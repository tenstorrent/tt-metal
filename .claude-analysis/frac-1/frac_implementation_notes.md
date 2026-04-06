# frac Implementation Notes

## Operation Definition
`frac(x) = x - trunc(x)` — returns the fractional part of x, preserving sign.

## Implementation Summary

The frac operation is implemented as a parameterless unary SFPU operation using IEEE 754 bit manipulation to extract the fractional part. The kernel uses three-case branching based on the debiased exponent:

1. **E < 0** (|x| < 1): The entire value is fractional, so `frac(x) = x`
2. **E >= 23**: The value is an exact integer (no fractional mantissa bits), so `frac(x) = 0`
3. **0 <= E < 23**: Mixed — create a mask `0xFFFFFFFF << (23 - E)` to zero out fractional mantissa bits, producing `trunc(x)`, then compute `frac(x) = x - trunc(x)`

Key SFPI primitives used:
- `sfpi::exexp(v)` — extract debiased exponent
- `sfpi::shft(vUInt, vInt)` — shift to create mantissa mask
- `sfpi::reinterpret<sfpi::vInt>` / `sfpi::reinterpret<sfpi::vFloat>` — bitwise reinterpret between float and int
- `v_if` / `v_endif` — predicated SIMD conditional branches

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/ttnn/operations/unary.py`
- `tt_metal/hw/sources.cmake`

## Reference Operations Used

1. **hardsigmoid** (most useful): Template for a parameterless unary op with no init constants. Provided the clean pattern for `calculate_frac()` structure, no-op init, and the `v_if/v_endif` conditional idiom.

2. **cbrt**: Provided the SFPI bit-manipulation patterns (`exexp`, `reinterpret<vInt>`, `reinterpret<vFloat>`) essential for the IEEE 754 exponent extraction and mantissa masking.

3. **softshrink**: Provided the three-case conditional pattern with a default result (`0.0f`) selectively overridden by `v_if/v_endif` branches.

4. **hardtanh**: Reinforced the standard ckernel_sfpu file structure and `v_if/v_endif` boilerplate.

5. **hardswish**: Showed the compute-intermediate-then-subtract pattern (similar to computing `trunc_val` then `x - trunc_val`).

## Deviations from Standard Patterns
- Uses nested `v_if` blocks (Case 3 is nested inside Case 1's else-branch via `exp >= 0`). This is safe because the CC stack supports nesting, as confirmed by the SFPU hardware model.
- No `APPROXIMATION_MODE` branching — the bit-manipulation algorithm is exact, no approximation variant needed.
- No `init` function needed — no programmable constant registers are used.

## Known Limitations
- bfloat16 precision: Since bfloat16 has only 7 mantissa bits (not 23), the exponent threshold for "exact integer" is lower. However, the kernel operates on FP32 values in the SFPU (bfloat16 is promoted to FP32 on load), so the 23-bit mantissa logic is correct.
- Very large floats (E >= 23) correctly return 0, matching IEEE 754 semantics where such values have no fractional bits.
- Subnormal numbers: `exexp` returns a debiased exponent; for subnormals this will be very negative, falling into Case 1 (result = x), which is correct since subnormals are always < 1 in magnitude.
