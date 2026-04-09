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

### Custom Compute Kernel
Created a dedicated compute kernel `eltwise_sfpu_rrelu.cpp` to avoid JIT compilation failures
caused by missing headers in the nuked repo (trigonometry.h, rpow.h, rdiv.h, fill.h, and
mul_int_sfpu.h's transitive dependency on ckernel_sfpu_mul_int32.h). The custom kernel only
includes the necessary headers for SFPU split includes.

### SfpuType Enum Restoration
The nuked repo had stripped the SfpuType enum to only 5 values, but the JIT-compiled
ckernel headers (ckernel_sfpu_comp.h, ckernel_sfpu_isinf_isnan.h, etc.) still reference
dozens of SfpuType members. Restored all required values to both wormhole_b0 and blackhole
llk_sfpu_types.h to enable JIT compilation.

### Reference Operations Used
- **threshold**: Conditional comparison pattern, parameter conversion
- **hardtanh**: Multi-parameter passing through UnaryWithParam, Python binding pattern
- **clamp**: v_if/v_elseif/v_endif branching
- **fill**: Basic SFPU kernel structure and Converter utility
- **dropout**: RNG pattern (for future training mode)

## Test Results
All 6 tests PASSED:
- test_rrelu_default[bfloat16]: PASSED
- test_rrelu_default[fp32]: PASSED
- test_rrelu_params[relu_like] (lower=0, upper=0): PASSED
- test_rrelu_params[identity_like] (lower=1, upper=1): PASSED
- test_rrelu_params[wide_range] (lower=0.01, upper=0.99): PASSED
- test_rrelu_params[default] (lower=0.125, upper=1/3): PASSED

## New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_rrelu.cpp`
- `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` (placeholder)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h` (placeholder)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rdiv.h` (placeholder)
- `tt_metal/hw/inc/api/compute/eltwise_unary/fill.h` (placeholder)
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`

## Modified Files
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (added RRELU to enum)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` (added to is_parametrized_type)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (registered in get_macro_definition, get_op_init_and_func_parameterized, get_compute_kernel_path)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (added rrelu C++ API function)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (added Python binding)
- `ttnn/ttnn/operations/unary.py` (registered golden function)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (added SFPU_OP_RRELU_INCLUDE)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (added rrelu + restored missing SfpuType values)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (added rrelu + restored missing SfpuType values)

## Known Limitations
1. Training mode (per-element random slope) not implemented in SFPU kernel
2. Only evaluation mode is supported (deterministic slope = (lower + upper) / 2)

## Debug Log
1. Initial test run: JIT compilation failed - `trigonometry.h: No such file or directory`
   - Fix: Created placeholder headers for trigonometry.h, rpow.h, rdiv.h, fill.h
2. Second test run: JIT compilation failed - `ckernel_sfpu_mul_int32.h: No such file or directory`
   - Fix: Created dedicated `eltwise_sfpu_rrelu.cpp` compute kernel without mul_int_sfpu.h include
3. Third test run: JIT compilation failed - `SfpuType::equal_zero is not a member of SfpuType`
   - Fix: Restored all missing SfpuType enum values referenced by transitive headers
4. Fourth test run: JIT compilation failed - `SfpuType::isinf is not a member of SfpuType`
   - Fix: Added isinf/isnan/isfinite and all remaining SfpuType values
5. Fifth test run: ALL 6 TESTS PASSED
