# Prompt Optimize

Analyze and improve a draft prompt so it is specific, testable, and ready to run in this repo.

## Behavior

- Use the `prompt-optimizer` skill.
- Advisory mode only: do not execute the requested implementation.
- Return:
  1) diagnosis
  2) missing context
  3) recommended commands/skills
  4) optimized full prompt
  5) optimized quick prompt

## Requirements

- Match the user's language.
- Prefer repository conventions from `AGENTS.md` and `.cursor/rules/`.
- Include explicit acceptance criteria and verification steps.
- Add scope boundaries ("do not change ...").

## Input

$ARGUMENTS
