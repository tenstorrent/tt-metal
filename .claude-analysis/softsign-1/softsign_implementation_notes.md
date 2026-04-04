# Softsign Implementation Notes

## Math Definition
`softsign(x) = x / (1 + |x|)`

## Implementation Strategy

Softsign is implemented as a non-parametrized unary SFPU operation using the Newton-Raphson reciprocal from `ckernel_sfpu_recip.h`.

The SFPU kernel computes:
1. `abs(x)` using `sfpi::abs(v)` — absolute value intrinsic
2. `1 + |x|` using `sfpi::vConst1` hardware constant for 1.0f
3. `1 / (1 + |x|)` using `_sfpu_reciprocal_<2>(denom)` — 2-iteration Newton-Raphson reciprocal
4. `x * recip` — multiply original value by reciprocal

The init function calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()` to program the SFPU constant registers (vConstFloatPrgm0/1/2) with the reciprocal polynomial coefficients.

## Reference Operations Used

1. **hardsigmoid** (primary template): Provided the complete end-to-end file skeleton — ckernel, LLK wrapper, compute API header, `unary_op_utils` registration, `sfpu_split_includes` guard, and `activations.h` aggregation. All new files follow this exact pattern.

2. **cbrt**: Demonstrated `sfpi::abs()` usage for absolute value computation and the `calculate_*/init` function naming convention.

3. **sigmoid/silu** (conceptual reference): Informed the `x * f(x)` multiply structure and reciprocal init pattern. The actual implementation uses the Newton-Raphson `_sfpu_reciprocal_` from `ckernel_sfpu_recip.h` rather than LUT-based sigmoid.

## Deviations from Standard Patterns

1. **Reciprocal include**: The kernel includes `sfpu/ckernel_sfpu_recip.h` from the third-party LLK library. This is the standard Newton-Raphson reciprocal implementation. Other custom kernels in this worktree (hardsigmoid, hardtanh) don't need reciprocal and thus don't have this include.

2. **Init function with callback**: The LLK wrapper passes `softsign_init<APPROXIMATE>` as an init callback to `llk_math_eltwise_unary_sfpu_init`, unlike hardsigmoid which has no init callback. This is necessary to program the reciprocal polynomial constants.

## Known Limitations

- Uses 2 Newton-Raphson iterations for the reciprocal, which provides ~1 ULP accuracy for fp32 but may have slightly higher error for extreme input values
- The reciprocal uses the programmable constant registers (vConstFloatPrgm0/1/2), so softsign cannot be fused in a chain with operations that also use these registers (e.g., cbrt)
- Approximation mode (APPROXIMATION_MODE=true) is not explicitly handled differently — the reciprocal always uses 2 iterations

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
- `tt_metal/hw/sources.cmake`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`
