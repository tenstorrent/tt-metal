# Hardswish Implementation Notes

## Math Definition
`hardswish(x) = x * min(max(x + 3, 0), 6) / 6 = x * hardsigmoid(x)`

## Implementation Strategy
Hardswish is mathematically `x * hardsigmoid(x)`, where `hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)`. The SFPU kernel computes the hardsigmoid intermediate, then multiplies by the original input `x`. This avoids needing any sub-function calls (exp, reciprocal, etc.) and keeps the kernel entirely self-contained with simple arithmetic and clamping.

## Reference Operations Used
- **hardsigmoid** (most useful): The kernel is a direct extension of hardsigmoid. The ckernel, LLK dispatch, and API header files all follow the exact same structure. The only difference is the final store: hardsigmoid stores `result` directly, while hardswish stores `x * result`.
- **hardtanh**: Provided a secondary pattern for clamping with `v_if`/`v_endif` blocks.
- **softsign** and **selu**: Referenced for understanding the full abstraction layer structure (API header macros, LLK dispatch patterns, params dispatch).

## Deviations from Standard Patterns
None. The implementation follows the exact same pattern as hardsigmoid across all layers:
- No init function needed (no programmable constants)
- No parameters (non-parameterized op)
- Standard `VectorMode::RC` dispatch
- `ADDR_MOD_7` with all-zero increments (default)
- `APPROXIMATION_MODE` template parameter accepted but unused

## Known Limitations
- The operation inherits bfloat16 precision limitations from the hardsigmoid computation (the `x/6 + 0.5` intermediate).
- For very large `|x|`, the clamping ensures correct piecewise behavior (0 for x <= -3, x for x >= 3).

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (added `hardswish` to `SfpuType` enum)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (added `hardswish` to `SfpuType` enum)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (added `SFPU_OP_HARDSWISH_INCLUDE` guard)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (added `HARDSWISH` to `get_macro_definition` and `get_op_init_and_func_default`)
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` (updated `HARDSWISH` in `get_macro_definition` and `get_op_init_and_func_default` to use SFPU tile functions instead of returning empty)
- `ttnn/ttnn/operations/unary.py` (added golden function using `torch.nn.functional.hardswish`)
