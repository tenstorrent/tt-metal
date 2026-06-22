---
name: beautify
description: Improve model code quality without changing behavior.
disable-model-invocation: true
---

# Beautify

Refactor the code to improve readability, maintainability, and reduce complexity.

## Examples of things to consider

- Is inheritance structured correctly? Should some layers be removed or added?
- Is there debugging code present that should be removed?
- Is there dead code?
- Do variable and function names reflect their purpose?
- Make sure comments are present where needed and only where needed, are up to date, and describe the intention behind the code
- Make sure paths aren't hardcoded, or user/machine specific
- Consider whether functions are correct length. Should some be split into multiple ones? Maybe some should be merged/inlined?
- Should a given function be a free function, or should it be a method of some class?
- Are there parts of the code that should become a separate class?

This is not an exhaustive list. Other fixes that would improve code quality are also in scope.

Before making a change, consider its pros and cons.

The end goal is to refactor the code such that a senior engineer would consider it well structured, clean, maintainable, tasteful and beautiful.


## Working Rules

- Preserve behavior, including model performance and accuracy.
- Prefer deleting complexity over adding abstractions.
- Keep external contracts stable unless explicitly requested.

## Suggested Workflow

1. Review the code to identify cleanup opportunities. Read all of the relevant code first to get an overview, then do a second pass considering possible improvements.
2. If there are no refactor opportunities left, report so and stop executing this skill.
2. Locate the accuracy tests and performance benchmarks used for the model
3. Measure the baseline performance and accuracy. If you have problems running the commands that look environment related, like readiness scripts arguments not matching, fix them, but also call them out separately in your report. Make sure you apply these fixes before any refactoring changes.
4. Perform the refactor
5. Re-test to validate the refactor.

## Verification

A refactor is valid if:

- No accuracy regressions.
- No performance regressions.
- Existing model tests still pass.

## Reporting

Report a short description of the improvements made.
When a refactor was considered but decided against, explain why.
Compare before and after performance and accuracy numbers in a table.
