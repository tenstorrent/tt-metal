---
name: modularize
description: Refactor model code to use existing TTTv2 modules where possible and add model-local candidate modules for uncovered pieces.
disable-model-invocation: true
---

# Modularize

Refactor the model code to
- adopt existing TTTv2 modules from `models/common/modules` when an applicable module already exists
- for remaining uncovered layers, create model-local candidate modules under `new_modules/modules`
- if a similar module exists, but is not applicable, document why exactly.
- add tests for new candidate modules under `new_modules/tests/modules`.

## Working Rules

- Preserve behavior, model quality, and measured performance.
- Prefer existing module contracts before inventing new ones.
- When creating a new module, match the conventions of exisitng ones

## Suggested Workflow

1. Map current model blocks to available `models/common/modules` candidates.
2. Refactor compatible pieces first.
3. For unmatched pieces, add `new_modules/modules` candidates.
4. Add focused tests in `new_modules/tests/modules`.
5. Re-run model checks and compare against baseline.

## Verification

- No accuracy regressions.
- No performance regressions.
- Existing model tests pass.
- New candidate modules have focused tests.