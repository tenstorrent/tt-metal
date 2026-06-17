---
name: reorganize
description: Reorganize an optimized full TTNN model by removing decoder inheritance layering and rebuilding around TTTv2-style modules without regressing performance or accuracy. Use after optimized-full-model when the user asks for decoder flattening, module extraction to models/common/modules with model-local candidates in new_modules/modules, or maintainability-focused refactors with strict parity gates.
---

# Reorganize Optimized Full Model

Use this skill after `$full-model` and `$optimize` have already produced a fast, correct full-model implementation.

This is a structural refactor stage. The output must keep behavior and speed at parity with the starting optimized full-model baseline.

## Mandatory Scope

Taking the full model, with the fastest version of the decoder:
1. Create a version that does not inherit from other versions of the decoder (optimized from functional etc)
2. Refactor the model to be built out of TTTv2-style modules (models/common/modules). Use existing modules when possible. When not possible, create a new candidate TTTv2-style module, complete with tests.

First scope out the refactor, then perform it step by step. Model performance and accuracy MUST NOT BE DEGRADED. Run tests to verify.

## Phase 1: Scope Before Code Changes

Before editing runtime code, write a concrete plan in `models/autoports/<model>/doc/reorganize/work_log.md`:

- identify the current fastest decoder implementation and the exact files/classes in use;
- map inheritance and mixin layering currently used by decoder/model/generator;
- identify which responsibilities can move to existing `models/common/modules/*` modules;
- identify gaps requiring new module(s), with proposed API and test shape;
- define acceptance gates (same harness and workload for before/after).

Do not start refactoring until this scope is written and baseline metrics are captured.

## Baseline Gates (Capture First)

Collect baseline accuracy and performance from the current optimized full model before any structural change:

- prefill and teacher-forcing readiness metrics (top-1/top-5/top-100);
- token-out autoregressive qualitative/degeneracy checks;
- warmed TTFT and trace-verified decode tokens/sec/user;
- representative unit/integration tests already used by the model path.

Use identical workload/config for before/after comparisons.

## Phase 2: Stepwise Refactor

Apply small, testable commits-in-spirit (even if you do not commit in git):

1. **Standalone decoder extraction**
   - create decoder code that does not inherit from older decoder variants;
   - keep the same runtime contract for prefill/decode inputs and outputs;
   - preserve selected dtype/fidelity, cache behavior, and trace behavior.

2. **TTTv2 module adoption**
   - replace model-specific ad-hoc blocks with existing `models/common/modules` implementations where compatible;
   - preserve sharding/layout/program-config assumptions required by the optimized path;
   - avoid host fallback or hidden eager fallbacks while reorganizing.

3. **New module(s) when needed**
   - if no existing module fits, implement a new TTTv2-style module under `new_modules/modules/...`;
   - include focused unit tests under `new_modules/tests/modules/...`;
   - keep module config-first, weight-loading boundary clear, and prefill/decode contracts explicit.

After each step, rerun targeted tests before moving on.

## No-Regression Policy

Treat any measurable regression as stage-incomplete:

- accuracy below baseline acceptance bars;
- decode/perf regression beyond normal run noise;
- new fallback in the measured runtime path;
- broken trace replay or sampler/token-feedback contract.

If a step regresses, fix it before proceeding. Do not defer regressions to later stages.

## Required Verification

Minimum required checks after the final refactor:

- readiness prefill and teacher-forcing checks pass at baseline bars;
- autoregressive output remains non-degenerate and coherent;
- warmed TTFT and decode t/s/u are at parity with baseline (or improved);
- module/unit tests for any newly introduced module pass;
- model path tests and generation path tests pass;
- runtime fallback audit is still clean for measured path.

## Evidence To Leave

Record final evidence in:

- `models/autoports/<model>/doc/reorganize/README.md`
- `models/autoports/<model>/doc/reorganize/work_log.md`

Include:

- before/after table for accuracy and performance;
- inheritance removal summary (old vs new structure);
- module migration map (reused modules and newly created modules);
- exact commands and tests run;
- known limitations and follow-ups.
