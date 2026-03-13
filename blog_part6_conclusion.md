# Summary of Findings

This research examined how large language models represent cultural values across **66 countries**, testing three prompting strategies—**simple prompting**, **chain-of-thought**, and **web-search agents**—against **World Values Survey (WVS)** data. Our analysis reveals three fundamental insights about AI cultural representation.

------------------------------------------------------------------------

## 1. Performance Varies Dramatically by Country

Using binary classifiers trained on LLM-generated responses, we found that predicted probabilities reveal stark heterogeneity in representational quality. The model performs extremely well for some countries—achieving near-perfect classification accuracy for Western democracies such as **Canada**, **Australia**, and **Germany**—but consistently fails for others, particularly in the **Global South** and mixed middle-income regions.

This is not simply an issue of overall accuracy. With similar error rates across all communities (approximately **57–61%**), the key difference lies in **which questions the model gets wrong**. Some societies are captured with remarkable fidelity, while others are systematically misrepresented through coarse cultural stereotypes.

The uneven distribution suggests that **training data coverage**, rather than inherent cultural complexity, likely drives representational quality.

------------------------------------------------------------------------

## 2. Error Communities Persist Across Prompting Methods

When we analyzed country-level error patterns using **community detection algorithms**, a striking pattern emerged: countries cluster into the **same error communities regardless of prompting strategy**.

Across **simple prompting**, **chain-of-thought**, and **web-search agents**, the same regional clusters consistently appear:

-   Western democracies\
-   Chinese cultural sphere\
-   Global South\
-   Mixed middle-income countries

More remarkably, **each community exhibits nearly identical error profiles across all three methods**. The same survey items that fail under simple prompting continue to fail under chain-of-thought and web-search prompting.

For example:

-   **Western cluster:** 100% error on *immigration job priority* across all methods\
-   **Global South:** 100% error on *filial duty* across all methods\
-   **Chinese sphere:** 100% error on *work importance* and *religiosity* across all methods

This persistence suggests that errors stem from **shared underlying knowledge gaps rather than reasoning failures**.

-   **Chain-of-thought prompting** does not correct cultural biases—it simply articulates them more elaborately.\
-   **Web search agents** can improve factual accuracy for well-documented topics but cannot overcome fundamental misunderstandings of **lived cultural experiences**.

------------------------------------------------------------------------

## 3. Item-Level Effects Reveal Method-Specific Impacts

While community-level patterns remain stable, **item-level analysis** reveals that prompting strategies reshape how the model responds to specific questions.

Different methods shift both the **direction** and **magnitude** of prediction errors:

-   **Chain-of-thought prompting**
    -   Improves **normative reasoning** (policy questions, institutional rules)
    -   Degrades **subjective items** (trust, family values, personal well-being)
-   **Web search agents**
    -   Improve **factual, rule-based questions** (unemployment benefits, abortion laws)
    -   Worsen **culturally embedded perceptions** (life satisfaction, trust in strangers)

Crucially, these effects are **redistributive rather than corrective**. Enhanced prompting helps some countries while harming others on the same item.

For example:

-   Chain-of-thought improves **tax fraud predictions** for **17% of countries**
-   But **degrades performance for 21% of countries**

Prompting therefore does **not universally improve cultural representation**—it **changes how the model fails**.

------------------------------------------------------------------------

# A Framework for Evaluating Cultural Representation in AI

This research demonstrates a systematic framework for assessing how AI systems represent cultural diversity.

**Step 1 — Ground Truth Construction**\
Use validated cross-cultural survey data (e.g., **World Values Survey**) as benchmarks.

**Step 2 — Agent Elicitation**\
Test multiple prompting strategies to expose different knowledge layers.

**Step 3 — Performance Evaluation**\
Use classifiers to quantify representational quality at scale.

**Step 4 — Error Analysis**\
Apply community detection to reveal systematic bias patterns.

**Step 5 — Item-Level Diagnosis**\
Identify which cultural domains benefit or suffer from each prompting method.

This pipeline reveals not just **how accurate models are**, but **how they fail**, exposing the structure of their cultural knowledge and the limits of different enhancement strategies.

------------------------------------------------------------------------

# Revealing Patterns: Stereotypes, Paradoxes, and Blindspots

Our analysis uncovered three particularly striking patterns.

------------------------------------------------------------------------

## 1. Inverse Stereotypes

Communities exhibit **contradictory error profiles**: what one group gets right, another gets wrong.

Examples include:

-   **Western countries**
    -   0% error on *gender job equality*
    -   100% error on *immigration priorities*
-   **Chinese cultural sphere**
    -   100% error on *work importance*
    -   0% error on *gender political leadership*

These inverse patterns reveal how the model applies **different cultural templates to different regions**, reinforcing **contradictory stereotypes**.

------------------------------------------------------------------------

## 2. The “Principles vs. Practice” Paradox

Models excel at **abstract ideals** but fail on **lived realities**.

Examples include:

-   **Western countries**
    -   8% error on *“people should obey their rulers”*
    -   92% error on *“suicide is justifiable”*
-   **Global South**
    -   27% error on *“work is a duty to society”*
    -   100% error on *“making parents proud is a life goal”*

The model appears to grasp **high-level normative values**, but struggles to predict **how those values manifest in everyday cultural contexts**.

------------------------------------------------------------------------

## 3. Community-Specific Blindspots

Each error community reveals distinct representational failures:

-   **Western democracies**
    -   Cannot accurately predict progressive social attitudes (e.g., LGBTQ rights, assisted suicide)
-   **Global South**
    -   Underestimates the intensity of collectivism and the centrality of family obligations
-   **Chinese cultural sphere**
    -   Misses the importance of work ethic and patterns of secular modernization
-   **Mixed middle-income countries**
    -   Fails to capture volatility in institutional trust
-   **Eurasian cluster**
    -   Oversimplifies hybrid authoritarian–democratic dynamics

These patterns reveal not random noise but **systematic knowledge gaps embedded in how LLMs learn about human societies**.

------------------------------------------------------------------------

# Conclusion

The stakes are high. As AI systems increasingly **mediate cross-cultural communication**, **inform policy decisions**, and **shape global discourse**, understanding their cultural blindspots is not optional—it is essential.

This research provides both:

-   **A warning** about the depth of cultural misrepresentation in current models\
-   **A roadmap** for systematically diagnosing and addressing these failures