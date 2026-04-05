# atanh Implementation Notes

## Math Definition
atanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| < 1 (inverse hyperbolic tangent)

## Implementation Strategy
The implementation follows the softsign pattern (LLK dispatch layer with separate SFPU kernel file).

The SFPU kernel computes atanh using:
1. Compute numerator: 1 + x
2. Compute denominator: 1 - x
3. Compute reciprocal of denominator using `_sfpu_reciprocal_<2>()`
4. Multiply to get ratio: (1+x) / (1-x)
5. Compute natural log using `_calculate_log_body_no_init_()`
6. Multiply by 0.5

## Reference Operations Used
- **softsign**: LLK dispatch pattern, reciprocal usage, compute API structure
- **acosh/asinh** (trigonometry.h): Same inverse hyperbolic family, `_calculate_log_body_no_init_()` usage
- **log**: Core log implementation via Chebyshev approximation
- **cosh**: Compute API macros pattern
- **selu**: Conditional SFPU logic pattern

## Key Design Decisions
1. Used `_sfpu_reciprocal_<2>()` for division (2 Newton-Raphson iterations for fp32 precision)
2. Used `_calculate_log_body_no_init_()` from `ckernel_sfpu_log.h` for the natural log (same as acosh/asinh)
3. Init function calls `_init_sfpu_reciprocal_<>()` to set up reciprocal constants
4. No special boundary handling needed - the hardware handles edge cases naturally:
   - x = 0: atanh(0) = 0.5 * ln(1/1) = 0.5 * 0 = 0
   - x -> 1: ratio -> inf, ln(inf) -> inf
   - x -> -1: ratio -> 0, ln(0) -> -inf

## Deviations from Standard Patterns
None - follows the exact same pattern as softsign and the inverse hyperbolic functions in trigonometry.h.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h
tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/ttnn/experimental_loader/golden_functions.py

## Known Limitations
- Input must be in range |x| < 1 for mathematically valid results
- Precision may be limited by the 3rd-order Chebyshev approximation used in the log implementation
- For values very close to +/-1, numerical precision degrades as the ratio approaches infinity/zero
