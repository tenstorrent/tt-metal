# Prompt Depth

Use the `token-budget-advisor` skill to choose answer depth before responding.

## Behavior

- If the user already requested depth (brief/moderate/detailed/exhaustive), apply it directly.
- Otherwise, ask the user to choose:
  - 25% Essential
  - 50% Moderate
  - 75% Detailed
  - 100% Exhaustive
- Keep the chosen depth for follow-up answers unless changed.

## Input

$ARGUMENTS
