---
name: token-budget-advisor
description: >-
  Lets users choose response depth before answering when they ask for short,
  medium, detailed, or exhaustive output. Keeps responses aligned to the
  user's requested depth.
---

# Token Budget Advisor

Use when users explicitly ask to control answer length/depth.

## Trigger Signals

- "short version"
- "be brief"
- "give me the detailed version"
- "keep this concise"
- "exhaustive answer"
- "50% depth" / "quick summary first"

Do not trigger for auth/session/payment token questions.

## Workflow

1. Estimate complexity: SIMPLE, MEDIUM, COMPLEX.
2. Offer depth options before full answer if the user did not already specify one.
3. Answer at the selected depth.
4. Keep that depth preference for follow-ups unless user changes it.

## Depth Levels

- **25% Essential**: direct conclusion only
- **50% Moderate**: answer + minimal context + one concrete example
- **75% Detailed**: structured explanation + tradeoffs + alternatives
- **100% Exhaustive**: full deep dive

## Default Prompt To User

Use this style before answering when no level is given:

```
Choose response depth:
1) Essential (25%)
2) Moderate (50%)
3) Detailed (75%)
4) Exhaustive (100%)
```

If the user already says "brief", "detailed", "full", etc., skip this prompt and answer directly at that level.
