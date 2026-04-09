# RReLU Implementation Notes

## Operation Summary

**Name**: rrelu (Randomized Leaky ReLU)
**Math Definition**:
```
f(x) = x            when x >= 0
f(x) = a * x        when x < 0
```
Where:
- Eval mode: `a = (lower + upper) / 2`
- Train mode: `a ~ Uniform(lower, upper)`, sampled independently per element

**Parameters**: lower (float, default=0.125), upper (float, default=1/3), training (bool, default=False)

## New Files

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

## Modified Files

- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — added `SFPU_OP_RRELU_INCLUDE` guard
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — added `SfpuType::rrelu` + missing LLK SfpuType entries
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` — added `SfpuType::rrelu` + missing LLK SfpuType entries
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — added `UnaryOpType::RRELU`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — added `RRELU` to `is_parametrized_type()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — added `RRELU` to `get_macro_definition()` and `get_op_init_and_func_parameterized()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — added `rrelu()` function declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` — added `rrelu()` function implementation
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — added Python binding for `ttnn.rrelu`
- `ttnn/ttnn/operations/unary.py` — added golden function using `torch.nn.functional.rrelu`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` — removed broken includes for nuked operations (trigonometry.h, rpow.h, rdiv.h, fill.h)

## Reference Operations Used

1. **atanh** (most useful): Provided the pattern for programmable constant registers (`vConstFloatPrgm0/1/2`), init function with parameter loading, and LLK wrapper structure with `_llk_math_eltwise_unary_sfpu_params_` dispatch.

2. **hardshrink** (useful for parameterized pattern): Showed how parameterized operations pass parameters via `UnaryWithParam` with multiple float params, and the `is_parametrized_type()` registration.

3. **swish**: Provided the pattern for `v_if/v_endif` predicated execution for sign-based branching, which rrelu uses for the `x < 0` condition.

4. **frac**: Showed the standard non-parameterized registration pattern, including `get_macro_definition` -> `SFPU_OP_*_INCLUDE` split-include guard chain.

5. **sinh**: Confirmed the standard LLK wrapper pattern and the identical WH/BH kernel approach.

## Architecture Decisions

### Parameter Passing
RReLU has 3 parameters (lower, upper, training). These are passed as a 3-element float vector via `UnaryWithParam{UnaryOpType::RRELU, {lower, upper, training_flag}}`.

In the dispatch layer (`get_op_init_and_func_parameterized`), the float parameters are bit-cast to `uint32_t` hex literals embedded in the init/func strings:
- `rrelu_tile_init(0x3e000000u, 0x3eaaaaabu);` — lower and upper as hex floats
- `rrelu_tile(0, 0u);` — dst index and training flag as uint literal

### SFPU Kernel Design
- **Eval mode**: Uses `vConstFloatPrgm2` (precomputed midpoint) for a single multiply on negative inputs. Very efficient — just one `v_if` branch and one multiply.
- **Training mode**: Generates per-element pseudo-random slopes using a xorshift PRNG. A namespace-scoped scalar state advances per iteration, and per-lane diversity comes from XORing with input element bits. The random value is mapped to [0,1) via IEEE 754 mantissa extraction, then scaled to [lower, upper].
- Uses `__builtin_memcpy` for aliasing-safe float/uint32_t conversion (avoids `-Werror=strict-aliasing`).

### Pre-existing Codebase Issues Fixed
- Removed broken unconditional includes from `eltwise_sfpu.cpp` (trigonometry.h, rpow.h, rdiv.h, fill.h) that referenced nuked files
- Added missing `SfpuType` enum entries required by third-party LLK template specializations (comparison ops, integer ops, etc.)

## Known Limitations

1. **Training mode PRNG**: The pseudo-random number generator is deterministic and seeded from input bits + a global counter. For the same input tensor and same invocation sequence, it will produce the same slopes. This is acceptable for a hardware implementation but differs from PyTorch's true PRNG behavior.

2. **Training mode slope range**: Due to bfloat16 precision and the mantissa-extraction approach, the generated slopes may not cover the full [lower, upper] range with uniform distribution. The actual observed range in testing is approximately [0.126, 0.331] vs the theoretical [0.125, 0.333].

3. **No backward operation**: Only the forward pass is implemented. A backward pass (`rrelu_bw`) would need to track which slope was used per element during training mode.
