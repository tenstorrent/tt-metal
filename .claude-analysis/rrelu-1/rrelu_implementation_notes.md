# RReLU Implementation Notes

## Overview
Implemented RReLU (Randomized Leaky ReLU) as a new unary SFPU operation in tt-metal.
Currently implements evaluation mode only (deterministic slope).

## Math Definition
```
output = x           if x >= 0
output = slope * x   if x < 0
where slope = (lower + upper) / 2  (evaluation mode)
```

Default parameters: lower=0.125 (1/8), upper=1/3

## Architecture Decisions

### Evaluation Mode Only (for SFPU kernel)
The SFPU kernel implements deterministic evaluation mode only. The slope is pre-computed
on the host as `(lower + upper) / 2` and passed as a single uint32_t parameter (bit-cast float).
Training mode (per-element random slope) would require RNG on the SFPU, which adds complexity
and is deferred.

### Parameter Passing
- The C++ API (`ttnn::rrelu()`) takes `lower` and `upper` as float parameters
- The slope is pre-computed on the host: `slope = (lower + upper) / 2.0f`
- The slope is passed through `UnaryWithParam{UnaryOpType::RRELU, slope}` as a single float parameter
- In `get_op_init_and_func_parameterized()`, the float slope is bit-cast to uint32_t for the
  SFPU kernel call: `rrelu_tile(idst, 0x...u)`
- In the SFPU kernel, `Converter::as_float(slope_param)` recovers the float value

### SFPU Kernel Design
Uses SFPI C++ API (not raw TTI instructions) for clarity and maintainability:
- `sfpi::vFloat slope = Converter::as_float(slope_param)` - parameter recovery
- `v_if(v < 0.0f) { v = v * slope; } v_endif;` - conditional scaling
- `#pragma GCC unroll 8` for loop unrolling
- Standard iteration pattern with `sfpi::dst_reg[0]` / `sfpi::dst_reg++`

### Reference Operations Used
- **threshold**: Conditional comparison pattern, parameter conversion
- **hardtanh**: Multi-parameter passing through UnaryWithParam, Python binding pattern
- **clamp**: v_if/v_elseif/v_endif branching
- **fill**: Basic SFPU kernel structure and Converter utility
- **dropout**: RNG pattern (for future training mode)

## New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

## Modified Files
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (added RRELU to enum)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` (added to is_parametrized_type)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (registered in get_macro_definition, get_op_init_and_func_parameterized)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (added rrelu C++ API function)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (added Python binding)
- `ttnn/ttnn/operations/unary.py` (registered golden function)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (added SFPU_OP_RRELU_INCLUDE)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (added rrelu to SfpuType)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (added rrelu to SfpuType)

## Known Limitations
1. Training mode (per-element random slope) not implemented in SFPU kernel
2. Only evaluation mode is supported (deterministic slope = (lower + upper) / 2)

## Build Status
Build successful.
