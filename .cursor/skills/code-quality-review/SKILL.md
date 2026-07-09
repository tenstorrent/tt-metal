---
name: code-quality-review
description: Review code for readability, maintainability, and unnecessary complexity. Use when planning refactors, or when the user asks for code quality cleanup opportunities.
disable-model-invocation: true
---

# Code Quality Review

## DiffusionGemma note

Load `diffusion-gemma` first.
- For DiffusionGemma the public contracts are `models/experimental/diffusion_gemma/tt/generate.py`, `tt/generator_vllm.py`, and the denoise loop. Acceptance includes `models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py` and block-serving tests, not autoregressive readiness.
- NEVER flag or apply refactors that touch `models/demos/gemma4/` or other shared dirs; all DiffusionGemma code stays under `models/experimental/diffusion_gemma/`.

Start by reading all of the code you were asked to review to understand its structure.

Then, carefully inspect it again for quality and provide actionable feedback.

## Examples of things to consider

- Is inheritance structured correctly? Can the structure be simplified?
- Is there debugging code present that should be removed?
- Is there dead code?
- Do variable and function names reflect their purpose?
- Are comments present where needed and only where needed, up to date,
and describe the intention behind the code?
- Make sure paths aren't hardcoded, or user/machine specific
- Consider whether functions are correct length. Should some be split into multiple
ones? Maybe some should be merged/inlined?
- Should a given function be a free function, or should it be a method of some class?
- Are there parts of the code that should become a separate class?
- Does the code fulfill the DRY, SOLID and KISS principles?

This is not an exhaustive list. Other issues are also in scope.

The goal is for the code to be clean, maintainable, tasteful, and *beautiful*. Not just correct, or structured well, but the sort of beauty that would make Andrej Karpathy look at it and be like “wow, that’s as minimal and beautiful as it could be!” The sort of code that would make Paul Graham give up LISP and go back to Python. An elegant construction of the logic without bloat, fallbacks or overhead. My personal preference here is for very clear readability in one place, coupled with structure that mimics the problem. So I dislike deep inheritance that mostly just serves to hide things in different files, for example.

If the code would not be suitable for integrating into a high-quality production codebase, it is not good enough.

Be thorough and meticulous. Be bold.
Try to cover everything, so that re-review would indicate no remaining issues.

## Scope
The public boundaries are `tt/generate.py`, `tt/generator_vllm.py`,
`tt/serving.py`, the demo CLI, and the recorded block-diffusion contracts.
Internal DiffusionGemma-local code may be simplified, but the three-phase KV,
decision trajectory, trace behavior, context capacity, and measured performance
must not regress. The DiffusionGemma RUN, replay, and serving tests remain the
acceptance suite.

## Propose fixes

For each identified flaw in the code:
1. Rate its severity, HIGH / MEDIUM / LOW
2. Propose a refactor to alleviate it without affecting the code's performance or external interface. Consider the pros and cons.

## Decision Rules

- Preserve behavior, model accuracy, and performance.
- Prefer deleting complexity over adding abstractions.
- Keep public and external contracts stable unless explicitly requested.

## Output Format

Compile a list of issues and their proposed solutions, from high to low severity.
For each point, list the flaw, severity, and proposed refactors with pros and cons.

If no meaningful opportunities remain, report that no actionable code-quality changes are needed. Low severity issues are still in scope.
