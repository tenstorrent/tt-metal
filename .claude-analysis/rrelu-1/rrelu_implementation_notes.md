# Implementation Notes: rrelu

## Math Definition
RReLU(x) = x if x >= 0, a*x if x < 0
- Training mode: a ~ Uniform(lower, upper), where a is randomly sampled per element
- Eval/inference mode: a = (lower + upper) / 2 (deterministic)
- Default parameters: lower = 1/8 = 0.125, upper = 1/3 ~ 0.3333

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py

## Design Decisions

### Reference Operations Used
- **prelu_sfpu_analysis**: Most useful for the eval-mode kernel -- the conditional multiply pattern (`v_if(v < 0.0f) { v = v * slope; } v_endif`) is identical to prelu. The SFPI abstraction layer pattern (vFloat, dst_reg, v_if/v_endif) was directly reused.
- **rand_analysis**: Essential for training-mode PRNG random number generation. The technique of reading the PRNG counter via `TTI_SFPMOV(0, 9, LREG3, 8)`, clearing sign, setting exponent to 127, subtracting 1.0 to get [0,1), then scaling with SFPMAD was adopted exactly from the rand kernel.
- **leaky_relu_analysis**: Provided the raw TTI instruction pattern for CC-guarded multiply on negative lanes: SFPLOAD -> SFPSETCC -> SFPMUL -> SFPENCC -> SFPSTORE. Used for the training-mode path.
- **selu_analysis**: Provided the 2-parameter registration pattern through all abstraction layers (LLK dispatch with custom init, compute API header calling LLK directly, nanobind with `bind_unary_composite_floats_with_default`). Extended to 3 parameters for rrelu.
- **dropout_analysis**: Provided the PRNG seeding pattern (`init_prng_seed(seed)` in the init function).

### Dual-Path Kernel Design
The SFPU kernel (`ckernel_sfpu_rrelu.h`) contains two distinct execution paths selected by a runtime `if` on the `training_uint` parameter:
- **Eval path**: Pure SFPI abstractions (vFloat, v_if/v_endif) matching the prelu pattern. The fixed slope is computed as `lower + range * 0.5`. Loop is `#pragma GCC unroll 8` for full unrolling.
- **Training path**: Raw TTI instructions for PRNG access + CC-guarded multiply. Loop is `#pragma GCC unroll 0` (no unrolling) because the PRNG read has side effects and register pressure is higher with LREG0-3 all in use.

### Precomputed Range Parameter
Instead of passing `lower` and `upper` directly to the SFPU kernel, the host-side `get_op_init_and_func_parameterized()` precomputes `range = upper - lower`. This avoids a floating-point subtraction on the SFPU, which would require either:
- Negating a register (complex in raw TTI without a dedicated negate instruction)
- Mixing SFPI and TTI instructions (fragile due to register allocation conflicts)

The 3 params passed to the kernel are: `lower`, `range`, `training_flag`.

### PRNG Seeding
The PRNG is seeded with a fixed seed (0) during `rrelu_tile_init()` via the LLK dispatch init function. This is a known limitation: the standard `UnaryProgramFactory` does not support passing per-tile seeds. Different cores will produce different random sequences because the PRNG LFSR state diverges after seeding, but tiles processed by the same core will share the PRNG state progression.

### Parameter Encoding
The `training` parameter (Python `bool`) is converted to `float` (1.0 for True, 0.0 for False) in `unary.cpp`, then bitcast to `uint32_t` in `unary_op_utils.cpp`. The kernel checks `training_uint != 0` to select the execution path (0x3f800000 for True, 0x00000000 for False).

### Wormhole/Blackhole Parity
Both architecture implementations are identical. The TTI instructions in the training path use `ADDR_MOD_3` (hardcoded literal `3`) for SFPLOAD/SFPSTORE, which maps to ADDR_MOD_7 on Wormhole via the addr mod base remapping and directly to ADDR_MOD_3 on Blackhole. Both result in zero auto-increment, which is the standard behavior for SFPU operations.

## Known Limitations
- **Fixed PRNG seed**: Training mode uses seed=0 for all invocations. True random behavior would require a custom program factory that passes per-core seeds.
- **No per-tile seed variation**: All tiles processed by the same core share the same PRNG state progression, so the random slopes are deterministic given the processing order.
- **BFloat16 rounding**: The eval-mode path uses SFPI abstractions which handle BFloat16 rounding automatically. The training-mode path uses raw TTI SFPSTORE with format mode 0 (default/BFloat16), which should handle rounding correctly for BFloat16 inputs.
- **PRNG quality**: The hardware PRNG is a 32-bit LFSR with period 2^32-1. The random slopes are uniform in [lower, upper) but with limited randomness quality compared to software PRNGs.
