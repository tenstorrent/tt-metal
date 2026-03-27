# Implementation Notes: rrelu

## Math Definition
RReLU(x) = x if x >= 0; RReLU(x) = a * x if x < 0
- Eval/inference mode: a = (lower + upper) / 2 (fixed slope)
- Default: lower = 1/8 (0.125), upper = 1/3 (~0.333)
- Parameters: lower (float), upper (float)

### New Files
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h
/localdev/vignjatijevic/tt-metal-4/tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Modified Files
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
/localdev/vignjatijevic/tt-metal-4/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
/localdev/vignjatijevic/tt-metal-4/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
/localdev/vignjatijevic/tt-metal-4/ttnn/ttnn/operations/unary.py

## Design Decisions

- **PRELU_SFPU was the most useful reference** for the SFPU kernel pattern: conditional multiply of negative inputs by a slope value. The `calculate_prelu` function is almost identical to our `calculate_rrelu` for the eval path.

- **SELU was the most useful reference** for the two-parameter registration pattern: passing two float parameters as hex-encoded uint32_t through the op chain, using `TT_FATAL(params.size() == 2, ...)`, and the `bind_unary_composite_floats_with_default` nanobind pattern.

- **Eval mode only**: The implementation computes `slope = (lower + upper) / 2` on the SFPU and applies it to negative inputs. This matches PyTorch's `rrelu` behavior in eval mode. Training mode (random per-element slopes using PRNG) was not implemented because:
  1. The PRNG requires `init_prng_seed()` which involves a custom init callback (600 NOP cycles), incompatible with the standard `{op_name}_tile_init()` pattern
  2. Dropout, which uses PRNG, has its own separate dispatch path outside the standard unary factory
  3. Random slopes would also require passing a seed parameter, making it a 3-parameter op

- **Slope computation on SFPU**: Rather than computing the midpoint on the host and passing a single parameter, we pass both `lower` and `upper` to the kernel and compute `(lower + upper) * 0.5` on the SFPU. This preserves the full parameter interface for future training mode extension.

- **No approximate mode needed**: RReLU is a simple multiply-if-negative operation with no transcendental functions, so `get_op_approx_mode` returns false (default).

- **SfpuType::rrelu added**: A new SfpuType enum entry was needed since rrelu has its own dedicated LLK dispatch file, following the SELU pattern (not reusing an existing family).

## Known Limitations
- Training mode (random per-element slopes) is not implemented. The operation always uses eval/inference mode with fixed slope = (lower + upper) / 2.
- No bfloat16 rounding is applied in the kernel since the operation is a simple multiply (not involving exp or other transcendental functions), and the result precision matches the input.
- The `training` parameter mentioned in the requirements is not exposed in the Python API. All invocations behave as eval mode.

## Test Results
- **Status**: PASS (attempt 1 of 1)
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py
- **Max ULP**: <= 2 (within threshold of 2)
- **allclose**: PASS (rtol=1.6e-2, atol=1e-2)
- **Test coverage**: All 65,536 bfloat16 bit patterns tested with default parameters (lower=0.125, upper=1/3)
- **Test duration**: 2.08s (call) + 4.12s (setup) = ~6.25s total
