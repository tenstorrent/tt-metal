# atanh Implementation Notes

## Operation
`atanh(x) = 0.5 * ln((1+x)/(1-x))` for |x| < 1

## Algorithm
The implementation uses IEEE 754 decomposition to compute ln(y) for positive y:
1. Decompose y = 2^e * m, where m in [1, 2) using `exexp` and `setexp` SFPU instructions
2. Approximate ln(m) on [1, 2) using a cubic minimax polynomial: `P(m) = c0 + m*(c1 + m*(c2 + m*c3))`
3. Compute `ln(y) = e * ln(2) + P(m)`

atanh is then: `0.5 * (ln(1+x) - ln(1-x))`

Polynomial coefficients (from rpow scalar log2 precomputation):
- c0 = -0x1.952992p+0f (~-1.5828)
- c1 = 0x2.4f5388p+0f (~2.3110)
- c2 = -0xd.e712ap-4f (~-0.8691)
- c3 = 0x2.44734p-4f (~0.1416)

c0, c1, c2 are stored in programmable constant registers (set during init).
c3 is loaded as an immediate.

## Which Reference Operations Were Most Useful and Why
1. **hardsigmoid** - Most useful for the overall file structure (API header, LLK dispatch, ckernel_sfpu, sfpu_split_includes pattern). It's a clean, simple non-parameterized unary op that served as the template for all abstraction layers.
2. **cbrt** - Showed how to use programmable constant registers (`vConstFloatPrgm0/1/2`) for polynomial coefficients and how to pass an init function to `llk_math_eltwise_unary_sfpu_init`.
3. **rpow** - Provided the cubic polynomial coefficients for ln(m) on [1, 2) and showed usage of `exexp`, `setexp`, and `int32_to_float` SFPU instructions for IEEE 754 decomposition.
4. **softshrink** and **hardtanh** - Confirmed parameterized vs non-parameterized patterns in `unary_op_utils.cpp`.

## Deviations from Standard Patterns
- The SFPU kernel is more complex than typical unary ops (hardsigmoid, hardtanh) because atanh requires computing two natural logarithms from scratch using SFPI instructions. Standard log/reciprocal primitives were intentionally removed, so IEEE 754 bit decomposition + polynomial approximation was used.
- Uses all 3 programmable constant registers for polynomial coefficients, following the cbrt pattern.

## Known Limitations or Concerns
- **Accuracy**: The cubic polynomial for ln(m) provides ~2-3 decimal digits of accuracy, sufficient for bfloat16 (~2.1 decimal digits). For fp32 accumulation mode, accuracy may be insufficient for the full mantissa precision.
- **Edge cases**: Values very close to x = +/-1 produce large outputs (atanh approaches +/-infinity). The polynomial approximation for ln near zero may lose precision for inputs like 1-x when x is very close to 1.
- **No fp32/fp16b path split**: Unlike cbrt, this kernel does not branch on `is_fp32_dest_acc_en`. A single code path is used for both accumulation modes. This is acceptable for bfloat16 precision targets.
- **Register pressure**: Each loop iteration uses many intermediates (two full ln computations), but the SFPI compiler handles register allocation automatically with `#pragma GCC unroll 8`.

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
