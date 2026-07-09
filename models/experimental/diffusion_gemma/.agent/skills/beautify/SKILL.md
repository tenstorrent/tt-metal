---
name: beautify
description: Improve model code quality without changing behavior.
disable-model-invocation: true
---

# Beautify

## DiffusionGemma note

Load `diffusion-gemma` first.
- The DiffusionGemma public contracts are `tt/generate.py`, `tt/generator_vllm.py`, `tt/serving.py`, and the denoise loop. Acceptance is the DiffusionGemma RUN/replay/serving suite. Measure diffusion decisions and traced performance before/after.
- NEVER refactor `models/demos/gemma4/` or shared dirs; keep all changes under `models/experimental/diffusion_gemma/`.

Refactor code to improve readability and maintainability while preserving behavior.

## Working Rules

- Preserve model external interface, performance and accuracy.
- Prefer deleting complexity over adding abstractions.

## Scope
Refactor only DiffusionGemma-local implementation details. Preserve the public
generation, serving, three-phase KV, context, decision-fidelity, and performance
contracts. Never treat existing tests as permission to restructure shared
Gemma-4 code.

## Suggested Workflow

1. Use Claude Code's Task/Agent mechanism to run `code-quality-review` on the target code in a fresh read-only subagent. That skill contains details on the scope and types of refactors expected.
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
