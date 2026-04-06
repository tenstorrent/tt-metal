# sinh Implementation Notes

## Operation
`sinh(x) = (exp(x) - exp(-x)) / 2`

Implemented as: `(2^(x * log2(e)) - 2^(-x * log2(e))) / 2` using the exp_21f algorithm.

## Reference Operations Used
- **rpow** (most useful): Provided the exp_21f algorithm for computing `2^z`, which is the core building block. The algorithm from Moroz et al. 2022 uses `addexp`, `exexp`, `exman9`, and polynomial refinement to compute 2^z efficiently on the SFPU.
- **hardsigmoid**: Provided the template for a non-parameterized unary operation — the LLK wrapper pattern, API header structure, and dispatch wiring in `unary_op_utils.cpp`.
- **softshrink**: Confirmed the parameterized dispatch pattern (not needed for sinh since it has no parameters), and validated the `sfpu_split_includes.h` conditional include mechanism.

## Implementation Strategy
1. Extracted the exp_21f algorithm from rpow into a reusable `exp_21f<APPROXIMATION_MODE>()` helper function within the sinh kernel header.
2. The SFPU kernel computes `exp(x)` and `exp(-x)` as two separate `exp_21f` calls with `z_pos = x * log2(e)` and `z_neg = -z_pos`.
3. Both z values are clamped to >= -127.0 to prevent underflow in the exp_21f algorithm.
4. Final result is `(exp_pos - exp_neg) * 0.5` with explicit bfloat16 rounding via `float_to_fp16b`.

## Deviations from Standard Patterns
- **exp_21f as inline helper**: Unlike rpow which has the exp_21f code inline in the main loop, sinh factors it into a separate templated function `exp_21f<APPROXIMATION_MODE>()` to avoid code duplication (it's called twice per element).
- **No special case handling**: Unlike rpow which handles base==0, base<0 etc., sinh has no domain restrictions requiring special cases. The clamping at -127 handles the underflow case naturally.
- **`#pragma GCC unroll 0`**: Used to prevent the compiler from unrolling the main loop, since each iteration is heavy (two exp_21f calls). This matches the rpow pattern.

## Known Limitations
- **Accuracy**: The exp_21f algorithm provides ~16-20 bits of precision, adequate for bfloat16 (8-bit mantissa). For very large |x| (beyond ~9), sinh values grow rapidly and may overflow bfloat16 range.
- **Performance**: Each element requires two exp_21f computations (~28 SFPU instructions per element), making this roughly 2x the cost of a single exponential operation.

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
- `tests/ttnn/unit_tests/operations/eltwise/unary/test_sinh.py`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/ttnn/operations/unary.py`
