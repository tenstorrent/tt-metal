# Softcap Implementation Notes

## Operation Definition
`softcap(x, cap) = cap * tanh(x / cap)` where `cap` is a positive float scalar (default 50.0).

## Implementation Strategy

Softcap is implemented as a **parameterized unary SFPU operation** following the hardtanh pattern for parameter passing and the sinh pattern for the SFPU kernel computation structure.

### SFPU Kernel Algorithm
The kernel computes `cap * tanh(x/cap)` using:

1. **z = x * inv_cap**: Division by cap is converted to multiplication by precomputed `1/cap` (encoded as BF16 on host).

2. **exp(z) and exp(-z)**: Computed using the `exp_21f` helper from the sinh kernel (Moroz et al. 2022 algorithm for 2^z). This is shared via `#include "ckernel_sfpu_sinh.h"`.

3. **tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))**: Division implemented via Newton-Raphson reciprocal:
   - Decompose denominator into mantissa `m in [1,2)` and exponent `e` using `sfpi::exexp`/`sfpi::setexp`
   - Linear initial estimate: `1/m ~ 1.4571 - 0.5*m` (~3.5 bits accuracy)
   - Two NR iterations: `y' = y * (2 - m*y)` (~14 bits, well above BF16's 8-bit mantissa)
   - Apply exponent: `rcp = inv_m * 2^(-e)` where `2^(-e) = setexp(1.0, 127-e)`

4. **Small-z override**: For `|z| < 0.5`, Taylor series `tanh(z) ~ z - z^3/3` avoids catastrophic cancellation in the exp subtraction.

5. **Final**: `result = cap * tanh_z`, rounded to BF16 via `float_to_fp16b`.

### Parameter Encoding
- Host precomputes `cap` and `1/cap`, encodes each as BF16 bit pattern (`uint32_t >> 16`)
- Parameters flow through SFPU_OP_CHAIN_0 string as uint32 literals
- Decoded in kernel via `sfpi::s2vFloat16b(param)` which broadcasts BF16 to SFPU vector register

## Reference Operations Used

### Most Useful: sinh (SFPU kernel pattern + exp_21f helper)
The sinh kernel provided:
- The `exp_21f` helper for computing 2^z (reused directly via include)
- The dual-path pattern: exp-based formula for moderate values, Taylor series for small values
- Loop structure: `#pragma GCC unroll 0`, `dst_reg[0]`/`dst_reg++`, BF16 rounding
- Init function pattern (empty, no programmable constants needed)

### Most Useful: hardtanh (parameterized dispatch pattern)
The hardtanh analysis provided:
- `is_parametrized_type()` registration pattern
- `get_op_init_and_func_parameterized()` case with host-side precomputation of derived params
- `s2vFloat16b(param)` for loading BF16-encoded scalars in SFPU kernel
- Function signature: `_calculate_hardtanh_(iterations, param0, param1, param2)`

### Supporting: swish, atanh
- swish: Showed the 5-layer stack structure and `REGISTER_UNARY_OPERATION` macro usage
- atanh: Showed `vConstFloatPrgm` preloading in init (not used for softcap, but informed the design decision to pass params as function arguments instead)

## Deviations from Standard Patterns

1. **Newton-Raphson reciprocal**: Unlike existing ops that avoid division entirely, softcap requires computing tanh which needs a ratio. Implemented NR reciprocal using IEEE float decomposition (`exexp`/`setexp`) + 2 refinement iterations. This is novel for this codebase.

2. **Cross-kernel include**: `ckernel_sfpu_softcap.h` includes `ckernel_sfpu_sinh.h` to reuse `exp_21f`. This creates a compile-time dependency but avoids code duplication.

3. **No saturation override for large |z|**: The exp-based formula handles saturation naturally (when |z| > ~4, exp(-|z|) underflows to 0, giving tanh ~ 1). No explicit saturation branch is needed, unlike the piecewise approach.

## Known Limitations

1. **BF16 precision of inv_cap**: The `1/cap` value is precomputed as float then truncated to BF16. For large cap values (e.g., 1000), `1/1000 = 0.001` has only ~3 significant bits in BF16, which may limit accuracy of `z = x * inv_cap`. This is inherent to the BF16 parameter encoding.

2. **Register pressure**: The kernel has many live variables (exp_pos, exp_neg, NR intermediates). The SFPI compiler manages LREG allocation and may spill to DEST. Performance impact is unknown without profiling.

3. **No APPROXIMATION_MODE branching**: Like sinh and atanh, the `APPROXIMATION_MODE` template parameter is accepted but does not alter the code path. Both modes execute the same algorithm.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
