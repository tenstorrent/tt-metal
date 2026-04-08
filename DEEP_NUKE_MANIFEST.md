# Deep Nuke Manifest

This branch (`vignjatijevic/sfpu-agent-codegen_deeply_nuked_for_rrelu`) is a controlled evaluation environment where SFPU unary operation implementations have been surgically removed so that AI code generators must implement them from raw SFPI instructions. The branch will be made orphan to prevent generators from recovering implementations through git history.

## Purpose

The SFPU generator evaluation (kernel_bench) measures whether AI agents can write correct SFPU kernels for Tenstorrent hardware. If pre-existing implementations or their mathematical primitives remain in the codebase, generators can copy them instead of reasoning about the math and hardware. Deep nuking removes these implementations across all abstraction layers, forcing generators to work from first principles.

## What Was Nuked

### Phase 1: Wave 4 Deep Nuke (commit `efdc0ad853`)

Removed **10 target operations** (originally from Waves 0-2) plus all **family primitives** they depend on. This was the foundation nuke for the Wave 4 evaluation round.

#### Target Operations Removed (Layers 2-5)

| Operation | Family | Dispatch | Compute API | Metal ckernel | Metal LLK | Tests |
|-----------|--------|----------|-------------|---------------|-----------|-------|
| hardsigmoid | Piecewise Linear | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| hardtanh | Piecewise Linear | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| hardswish | Piecewise Linear | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| softshrink | Piecewise Linear | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| cbrt | Bit-Manipulation | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| rpow | Bit-Manipulation | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| cosh | Exp-Composition | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| softsign | Log+Reciprocal | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| lgamma | Log+Reciprocal | removed | deleted | deleted (wh+bh) | deleted (wh+bh) | deleted |
| selu | Exp-Composition | removed | deleted | (was generated) | (was generated) | deleted |

Note: Enum values and REGISTER_UNARY_OPERATION macros were intentionally kept for these 10 ops so the build succeeds and generators can wire new implementations into the existing infrastructure.

#### Family Primitives Removed (Layer 1: tt_llk submodule)

The tt_llk submodule was pointed to commit `afd0b15d` ("Remove Family 1/3/4 SFPU primitives for Wave 4 evaluation") which deleted the core mathematical building blocks that target operations depend on:

**Family 1: Exponential-Composition** (used by swish, sinh, cosh, selu, softplus)
- `ckernel_sfpu_exp.h` (wh+bh+quasar) -- `_sfpu_exp_21f_bf16_`, `_calculate_exponential_*`, `_init_exponential_`
- `ckernel_sfpu_sigmoid.h` (wh+bh+quasar) -- `_calculate_sigmoid_`, uses exp
- `ckernel_sfpu_gelu.h` (wh+bh+quasar) -- `_calculate_gelu_*`, similar x*activation(x) pattern
- `ckernel_sfpu_tanh.h` (wh+bh+quasar) -- `_calculate_tanh_`, sigmoid via tanh
- `ckernel_sfpu_tanh_derivative.h` (wh+bh) -- tanh derivative
- `ckernel_sfpu_elu.h` (wh+bh) -- `_calculate_elu_`, uses exp
- `ckernel_sfpu_silu.h` (wh+bh+quasar) -- `_calculate_silu_`, uses exp+sigmoid

**Family 3: Logarithm + Reciprocal** (used by atanh, softsign, lgamma)
- `ckernel_sfpu_log.h` (wh+bh) -- `_calculate_log_body_*`
- `ckernel_sfpu_recip.h` (wh+bh+quasar) -- `_sfpu_reciprocal_*`
- `ckernel_sfpu_trigonometry.h` (wh+bh) -- `_calculate_asinh_`, `_calculate_acosh_`, sine/cosine Maclaurin series

**Family 4: Bit-Manipulation** (used by frac, cbrt, rpow, floor, ceil, trunc)
- `ckernel_sfpu_rounding_ops.h` (wh+bh) -- `_calculate_floor_`, `_calculate_ceil_`, `_calculate_trunc_`, `_calculate_round_`

**Collateral damage (stubbed, not deleted):**
- `ckernel_sfpu_exp2.h` -- depends on exp, implementation emptied
- `ckernel_sfpu_binary.h` -- POW/DIV/XLOGY use exp/log/recip, those function bodies removed
- `ckernel_sfpu_activations.h` -- CELU uses exp, specialization stubbed (empty body)

### Phase 2: Relu + ELU Family Nuke (commit `8c0af4489d`)

Extended the nuke to cover **7 additional operations** from the relu and ELU families. These were not part of the original Wave 4 evaluation but are being removed to create a clean environment for rrelu evaluation.

#### Operations Fully Removed (enum + registration + all layers)

| Operation | What Was Removed |
|-----------|-----------------|
| LEAKY_RELU | UnaryOpType enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER |
| PRELU_SFPU | UnaryOpType enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER, binary composite `prelu()` stubbed |
| RELU_MAX | UnaryOpType enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER |
| RELU_MIN | UnaryOpType enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER |
| ELU | UnaryOpType enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER |
| SELU | UnaryOpType enum value, REGISTER_UNARY_OPERATION, nanobind binding |
| CELU | UnaryOpType enum value (registration was already absent) |

Unlike the Phase 1 ops, these 7 operations had their enum values and registrations fully removed since they are not evaluation targets -- they are being nuked to prevent generators from learning relu/elu patterns.

#### Relu + ELU Primitives Removed (tt_llk submodule)

The tt_llk submodule was further modified (commit `b3bd59c8` on branch `vignjatijevic/sfpu-agent-codegen_deeply_nuked`):

- `ckernel_sfpu_relu.h` (wh+bh) -- stripped to only `_relu_max_body_` clamp helper (retained because hardsigmoid depends on it). Removed: `_calculate_lrelu_`, `_relu_max_impl_`, `_relu_max_`, `_relu_min_impl_`, `_relu_min_`, `is_supported_relu_type_v`
- `ckernel_sfpu_lrelu.h` (quasar) -- **deleted entirely** (leaky_relu, relu_min, relu_max for quasar)
- `ckernel_sfpu_activations.h` (wh+bh) -- removed CELU specialization (was already stubbed with empty body, now fully removed)
- `ckernel_defs.h` (wh+bh) -- removed `ActivationType::Celu` and `ActivationType::Elu` from enum
- `llk_sfpu_types.h` (test helper) -- removed: lrelu, relu_max, relu_min, elu, celu, prelu
- `sfpu_operations.h` (test helper) -- removed celu, relu_max, relu_min dispatch cases
- `ckernel_sfpu.h` (quasar umbrella) -- removed `#include "sfpu/ckernel_sfpu_lrelu.h"`

Note: ELU's `ckernel_sfpu_elu.h` and the exp primitive it depends on were already deleted in Phase 1.

### Phase 3: RELU/RELU6 Made Dysfunctional (commit `7f4ff0f050`)

RELU and RELU6 are packer-level operations (not SFPU), but their implementations were gutted to prevent generators from learning relu patterns:

- `ckernel_sfpu_relu.h` (wh+bh) -- emptied completely (just `#pragma once`)
- `ckernel_sfpu_activations.h` (wh+bh) -- hardsigmoid `apply()` body emptied (identity function), `_init_hardsigmoid_()` body emptied, `ckernel_sfpu_relu.h` include removed
- RELU/RELU6 registrations kept (symbols exist, fused activation paths in matmul/conv compile)
- RELU/RELU6 dispatch already removed in Phase 1 -- standalone `ttnn::relu()`/`ttnn::relu6()` throw at runtime

### Phase 4: Quasar Relu Nuke

Quasar `ckernel_sfpu_relu.h` still had a full relu implementation that was missed in Phase 3 (which only gutted wh+bh). Cleaned up:

- `ckernel_sfpu_relu.h` (quasar) -- emptied completely (just `#pragma once`)
- `ckernel_sfpu.h` (quasar umbrella) -- removed `#include "sfpu/ckernel_sfpu_relu.h"`
- `llk_defs.h` (quasar) -- removed `relu`, `lrelu`, `relumin`, `relumax` from `SfpuType` enum (dead values, no code references them)

### Build Fixes (commit `9b2679bc62`)

The deep nuke exposed pre-existing issues in `sources.cmake` and `activations.h`:
- `sources.cmake` referenced 4 deleted compute API headers (hardsigmoid.h, hardswish.h, hardtanh.h, softshrink.h) -- removed
- `activations.h` included deleted hardsigmoid.h and softsign.h -- cleaned
- `binary_composite_op.cpp` had unused parameter warnings in stubbed prelu() -- fixed

## What Survives

### Operations still functional on this branch
- RELU, RELU6 (registered but dispatch removed -- throw at runtime)
- All non-SFPU unary operations (abs, neg, recip, sqrt, etc.)
- Wave 3 generated ops: swish, frac, atanh, sinh (dispatch removed but enum/registration kept)
- Wave 0-2 ops: hardsigmoid, hardtanh, hardswish, softshrink, cbrt, rpow, cosh, softsign, lgamma (dispatch removed but enum/registration kept)

### tt_llk primitives still available
- `ckernel_sfpu_relu.h` (wh+bh+quasar) -- **emptied** (just `#pragma once`, gutted in Phases 3+4)
- `ckernel_sfpu_sqrt.h` -- square root
- `ckernel_sfpu_square.h` -- square
- `ckernel_sfpu_abs.h` -- absolute value
- `ckernel_sfpu_sign.h` -- sign
- `ckernel_sfpu_comp.h` -- comparisons
- `ckernel_sfpu_negative.h` -- negation
- `ckernel_sfpu_fill.h` -- fill constant
- `ckernel_sfpu_converter.h` -- float conversion utilities
- `ckernel_sfpu_load_config.h` -- SFPU config loading
- Hardware SFPI instructions (sfpi.h) -- the raw building blocks generators must use

### What generators must implement from scratch
Any operation that needs: exponential, logarithm, reciprocal, trigonometry, rounding/floor/ceil, sigmoid, tanh, gelu, elu, leaky_relu, relu_max, relu_min, or any composition thereof.
