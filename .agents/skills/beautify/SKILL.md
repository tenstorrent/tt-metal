---
name: beautify
description: Improve model code quality without changing behavior.
disable-model-invocation: true
---

# Beautify

## Scope

- remove unnecessary inheritance layers;
- remove superfluous debug machinery that is unlikely to be used in the future;
- remove dead code
- apply general taste and maintainability fixes.
- make sure comments are present where needed and only where needed, up to date, and describe the intention behind the code

The end goal is to refactor the code such that a senior engineer would consider it well structured, clean, maintainable and beautiful.

## Working Rules

- Preserve behavior, including model performance and accuracy.
- Prefer deleting complexity over adding abstractions.
- Keep external contracts stable unless explicitly requested.

## Suggested Workflow

1. Review the code to identify cleanup opportunities. If there are none left, report so and stop executing this skill.
2. Locate the accuracy tests and performance benchmarks used for the model
3. Measure the baseline performance and accuracy
4. Perform the refactor
5. Re-test to validate the refactor.

## Verification

A refactor is valid if:

- No accuracy regressions.
- No performance regressions.
- Existing model tests still pass.

## Reporting

Report a short description of the improvements made and before and after performance and accuracy numbers.
