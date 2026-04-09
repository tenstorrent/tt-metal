# RReLU Implementation Notes

## Operation Summary
RReLU (Randomized Leaky ReLU) in evaluation mode:
- `rrelu(x) = x` if `x >= 0`
- `rrelu(x) = slope * x` if `x < 0`
- where `slope = (lower + upper) / 2`

Parameters: `lower` (default 0.125), `upper` (default 1/3)

## Architecture
Standard parametrized unary SFPU operation using the SFPU_OP_CHAIN dispatch path through `eltwise_sfpu.cpp`. The slope is pre-computed on the host from `(lower + upper) / 2`, packed as FP16_B (bfloat16 bits in uint32_t), and embedded as a hex literal in the SFPU_OP_CHAIN_0 macro string. No runtime args needed.

## Reference Operations Used
- **frac**: Primary template for the standard unary SFPU wiring pattern (SFPU kernel, LLK wrapper, compute API, split include, host dispatch). Frac is the cleanest no-parameter example.
- **hardtanh**: Primary template for parametrized ops (2 float params, `is_parametrized_type`, `get_op_init_and_func_parameterized`, `s2vFloat16b` parameter loading in kernel, C++ API with explicit params, `unary_two_float_5param_to_6param_wrapper` nanobind).
- **swish**: Referenced for golden function pattern and nanobind registration.

## Deviations from Standard Patterns
- None. Followed standard parametrized unary SFPU patterns exactly.

## Known Limitations
- Only evaluation mode is implemented (deterministic slope). Training mode (stochastic per-element slope from Uniform(lower, upper)) is not supported because the SFPU does not have a suitable random number generator for this use case.
- The slope is quantized to bfloat16 precision when packed for the SFPU kernel.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
ttnn/ttnn/operations/unary.py
