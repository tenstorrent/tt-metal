# softshrink Implementation Notes

## Math Definition
softshrink(x, lambda) = x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise

## Implementation Summary
Implemented softshrink as a new SFPU unary operation with a single float parameter (lambda, default=0.5).

## Reference Operations Used
- **hardtanh** (BEST MATCH): Used as primary reference for piecewise branching with parameters via `v_if`/`v_endif` and bitcast float parameter passing
- **rpow**: Referenced for single-parameter `get_op_init_and_func_parameterized` pattern
- **hardsigmoid**: Referenced for piecewise clamping SFPU kernel structure
- **softsign**: Referenced for the complete 12-layer integration pattern

## Key Design Decisions
1. **Parameter passing**: Lambda is passed as a single `uint32_t` (IEEE 754 bitcast), following the rpow pattern. The `Converter::as_float()` utility reconstructs the float on the SFPU side.
2. **SFPU kernel logic**: Uses three-way branching (x > lambda, x < -lambda, else) with `v_if`/`v_endif`. Result initialized to 0.0f, then conditionally overwritten.
3. **Unroll pragma**: Uses `#pragma GCC unroll 8` since the kernel body is lightweight (no transcendentals, just comparisons and arithmetic).
4. **No init function needed**: Unlike selu or softsign, softshrink has no special initialization requirements (no reciprocal table, no exp init).

## New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_softshrink.py`

## Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added `softshrink` to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added `softshrink` to SfpuType enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added `SFPU_OP_SOFTSHRINK_INCLUDE` conditional
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Added SOFTSHRINK to `get_macro_definition()` and `get_op_init_and_func_parameterized()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added SOFTSHRINK to `is_parametrized_type()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added nanobind registration for `softshrink` with `lambd` parameter
- `ttnn/ttnn/operations/unary.py` -- Added golden function using `torch.nn.functional.softshrink`

## Known Limitations
- The SFPU kernel operates on bfloat16/float32 tiles. For very large lambda values, the threshold comparison is exact in float32 but subject to bfloat16 truncation in bf16 mode.
