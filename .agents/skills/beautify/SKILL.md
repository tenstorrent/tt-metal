---
name: beautify
description: Improve model code quality without changing behavior.
disable-model-invocation: true
---

# Beautify

Refactor code to improve readability and maintainability while preserving behavior.

## Working Rules

- Preserve model external interface, performance and accuracy.
- Prefer deleting complexity over adding abstractions.

## Scope
The goal is to have a model usable through vLLM and a demo script. Our interface to the world is through generator.py and generator_vllm.py's external contracts. Everything below this level is entirely within scope to restructure and rewrite at will, even if it has supporting tests - for example earlier versions of the decoder for models that only use the optimized multichip version.
You’re allowed and expected to change the API usage where appropriate to make things more elegant but neither correctness nor performance may regress.
The vLLM and readiness tests are our acceptance tests - they must still pass under the same conditions!

## Suggested Workflow

1. Use an intelligent subagent to run `$code_quality_review` on the target code to identify opportunities for improvement. That skill contains details on the scope and types of refactors expected.
2. If the previous step reports no actionable changes, report that and stop executing this skill.
3. Locate the accuracy tests and performance benchmarks used for the model
4. Measure the baseline performance and accuracy. If you have problems running the
commands that look environment related, like readiness scripts arguments not matching,
fix them, but also call them out separately in your report. Make sure you apply these
fixes before any refactoring changes.
5. Perform the refactor
6. Re-test to validate the refactor.

## Verification

A refactor is valid if:

- No accuracy regressions.
- No performance regressions.
- Existing model tests still pass.

## Reporting

Report a short description of the improvements made.
Preserve the subagent's review verbatim.
Preserve a commented copy of the subagent's review. For each suggestion, say if it was implemented. If it was considered but decided against, explain why.
Compare before and after performance and accuracy numbers in a table.
