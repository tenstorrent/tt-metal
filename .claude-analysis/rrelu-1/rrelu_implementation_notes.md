# RReLU SFPU Operation Implementation Notes

## Operation Definition
Randomized Leaky ReLU (RReLU):
- f(x) = x when x >= 0
- f(x) = a * x when x < 0
- Eval mode: a = (lower + upper) / 2
- Train mode: a ~ Uniform(lower, upper), sampled independently per element
- Default: lower=0.125, upper=1/3, training=False

## Implementation Summary

### Architecture
RReLU is implemented as a parameterized unary SFPU operation using the standard `UnaryProgramFactory` dispatch chain. It takes 3 parameters (lower, upper, training) passed through the `UnaryWithParam` mechanism.

### Eval Mode
Uses SFPI C++ abstractions for clean implementation:
- Precomputes slope = (lower + upper) / 2 once before the iteration loop
- For each element: if x < 0, x = x * slope; else passthrough

### Training Mode
Uses raw TTI instructions for hardware PRNG access (following the dropout pattern):
- Seeds PRNG once via `init_prng_seed()` with a static guard
- Generates random floats in [lower, upper] per element using:
  1. Hardware PRNG generates uint32 per lane via `TTI_SFPMOV(0, 9, LREG, 8)`
  2. `SFPSETEXP(127)` forces exponent to 127, creating float in [1.0, 2.0)
  3. `SFPSETSGN` clears sign bit
  4. Single SFPMAD maps [1.0, 2.0) to [lower, upper): `a = rand * range + offset`
     where range = upper - lower, offset = 2*lower - upper
  5. Conditional execution via `SFPSETCC` + CC-guarded `SFPMAD` applies `x * a` only for x < 0

### Parameter Encoding
Host-side (unary_op_utils.cpp) encodes parameters as:
- param0 = bit_cast<uint32_t>(lower)
- param1 = bit_cast<uint32_t>(upper)
- param2 = training flag (0 or 1)

The SFPU kernel receives these as uint32_t and reconstructs floats via `Converter::as_float()`.

## Reference Operations Used
1. **swish** - Primary reference for the full dispatch chain (API header -> LLK dispatch -> SFPU kernel). Used as template for all abstraction layer files.
2. **dropout** - Reference for hardware PRNG access patterns (TTI_SFPMOV with mod1=8, lreg_c=9), PRNG seed initialization, and conditional execution with raw TTI instructions.
3. **hardtanh** - Reference for parameterized operation pattern (multiple uint32 params, Converter::as_float usage, is_parametrized_type registration).
4. **threshold** - Reference for conditional execution pattern (SFPSETCC + CC-guarded operations + SFPENCC).
5. **clamp_tss** - Reference for Python binding pattern with multiple float parameters.

## Deviations from Standard Patterns
1. **Hybrid SFPI/TTI approach**: Eval mode uses clean SFPI abstractions; training mode uses raw TTI instructions for PRNG. This is necessary because SFPI has no abstraction for the hardware PRNG.
2. **Static PRNG guard**: Uses `static bool` to seed PRNG only once across all tile invocations. This avoids the 600-NOP overhead per tile and ensures different tiles get different random values.
3. **Fixed PRNG seed**: Uses a hardcoded seed (0xDEADBEEF). Different cores get the same seed, producing correlated random patterns. A production implementation would want per-core seeds via runtime args.

## Known Limitations
1. **Fixed PRNG seed**: All cores use the same seed, producing identical random sequences. For multi-core deployment, per-core seeds would be needed (similar to dropout's `DropoutMeshWorkloadFactory`).
2. **Training mode precision**: The random float generation uses the 23-bit mantissa of IEEE 754, which provides good but not perfect uniformity in bfloat16 (only 8 mantissa bits).
3. **No explicit PRNG re-seeding**: The PRNG state advances naturally but cannot be controlled per-invocation through the standard unary dispatch.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
