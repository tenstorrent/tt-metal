# lgamma Implementation Notes

## Operation
- **Name**: lgamma
- **Math Definition**: ln(|Gamma(x)|)
- **Parameters**: None
- **UnaryOpType**: LGAMMA (already existed in enum)

## Algorithm
Uses the Lanczos approximation with g=5 (Numerical Recipes coefficients), matching the existing composite implementation in `unary_composite_op.cpp`.

Formula:
```
lgamma(x) = 0.5*ln(2*pi) + (x - 0.5)*ln(x + 4.5) - (x + 4.5) + ln(series)
```
where:
```
series = 1 + 76.18/x - 86.51/(x+1) + 24.01/(x+2) - 1.23/(x+3)
```

The first 4 Lanczos coefficients are used (c5 and c6 are negligibly small for bfloat16 precision).

### SFPU Helpers Used
- `_sfpu_reciprocal_<1>` from `sfpu/ckernel_sfpu_recip.h` - 1 Newton-Raphson iteration (bfloat16 sufficient)
- `_calculate_log_body_no_init_` from `sfpu/ckernel_sfpu_log.h` - inline-constant variant (no programmable register conflict)

### Init Function
`lgamma_init()` calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()` to set up the 3 programmable constant registers (vConstFloatPrgm0/1/2) needed by the reciprocal function.

### Special Cases
- lgamma(1) = 0 and lgamma(2) = 0 are handled explicitly via `v_if` conditionals.
- Negative x values are not handled by the Lanczos formula (the approximation is only valid for x > 0). The reflection formula would be needed for full support but is too complex for SFPU.

## Reference Operations Used
- **hardsigmoid**: Most useful for the overall file structure pattern - simple no-parameter unary op with identical WH/BH implementations. Used as the template for API header, LLK dispatch, and registration patterns.
- **selu**: Useful for understanding how `_init_sfpu_reciprocal_` and `_calculate_exponential_piecewise_` are called from SFPU kernels. Showed the pattern for including `sfpu/ckernel_sfpu_exp.h` (which transitively includes recip).
- **cosh**: Showed how to include tt_llk helper functions (`sfpu/ckernel_sfpu_exp.h`).
- **cbrt**: Showed the programmable constant register pattern and `#pragma GCC unroll` usage.

## Deviations from Standard Patterns
- Uses `#pragma GCC unroll 0` (no unrolling) instead of `#pragma GCC unroll 8` because the lgamma kernel body is very large (4 reciprocal calls + 2 log calls + series accumulation + conditionals), and unrolling 8x would cause excessive code size.
- Includes two tt_llk helper headers (`sfpu/ckernel_sfpu_log.h` and `sfpu/ckernel_sfpu_recip.h`) whereas most simple ops only include `ckernel.h` and `ckernel_defs.h`. This is necessary because lgamma is a composite function requiring log and reciprocal building blocks.

## Known Limitations
1. **Positive x only**: The Lanczos approximation is valid for x > 0. Negative non-integer inputs would need the reflection formula (lgamma(x) = ln(pi/|sin(pi*x)|) - lgamma(1-x)), which requires a sin(pi*x) implementation not available as an SFPU helper.
2. **Precision**: Using `_sfpu_reciprocal_<1>` (1 Newton iteration) gives bfloat16-level precision. For float32 accuracy, `_sfpu_reciprocal_<2>` would be needed, but register pressure may be an issue.
3. **Performance**: The kernel is computation-heavy (~4 reciprocals + 2 logs per element). This is inherent to the lgamma function's complexity.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h
tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
