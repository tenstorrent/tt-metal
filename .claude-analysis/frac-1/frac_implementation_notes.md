# Implementation Notes: frac

## Math Definition
frac(x) = x - floor(x) -- the fractional part of x, always in [0, 1).

## SFPU Kernel Algorithm
The kernel implements frac using IEEE 754 bit manipulation:

1. Extract the debiased exponent E via `sfpi::exexp(v)`
2. Three cases:
   - **E >= 23**: The float has no fractional bits, result = 0
   - **E < 0**: |x| < 1, entire value is fractional. For positive x: result = x. For negative x: result = x + 1
   - **0 <= E < 23**: Build a bitmask to zero out fractional mantissa bits (trunc operation), compute diff = x - trunc(x). For positive x: result = diff. For negative non-integer x: result = diff + 1

3. The mask is computed as: `0xFFFFFFFF << (23 - E)` which zeroes the lower (23-E) mantissa bits.

## Reference Operations Used
- **hardswish**: Primary pattern reference (simple no-param unary, same file structure)
- **softsign**: Secondary pattern reference (LLK dispatch with custom init pattern)
- **cbrt**: SFPI bit manipulation patterns (reinterpret, shift, exponent extraction)

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h
- tt_metal/hw/inc/api/compute/eltwise_unary/frac.h

### Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/ttnn/operations/unary.py

## Pre-existing Registrations (no changes needed)
- `UnaryOpType::FRAC` already in enum (unary_op_types.hpp line 103)
- `REGISTER_UNARY_OPERATION(frac, FRAC)` already in unary.hpp line 154
- `bind_unary_operation<"frac", &ttnn::frac>` already in unary_nanobind.cpp

## Known Limitations
- The SFPI `shft` operation with vector shift amount may have performance implications
- Deep v_if nesting (3 levels) may impact SFPU throughput
