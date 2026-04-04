# Implementation Notes: hardtanh

## Math Definition
max(min_val, min(max_val, x)) where min_val=-1.0 (default), max_val=1.0 (default)

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h
tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py

## Design Decisions
- **Reference operations most useful**: relu6_analysis was the most useful reference because RELU6 (`min(max(x, 0), 6)`) is a special case of hardtanh. The relu_max kernel pattern of using v_if comparisons with Converter::as_float for parameter reconstruction was directly adapted. The hardsigmoid_analysis was also useful for understanding the SFPU dispatch chain through standard LLK macros.
- **SFPU kernel approach**: Created a clean two-comparison clamp kernel using direct `v_if(val < min_val)` and `v_if(val > max_val)` comparisons, rather than the existing tt_llk kernel's 3-parameter arithmetic trick (add/clamp/add/clamp/add). The direct approach is simpler, more readable, and avoids precision loss from chained FP16_B additions.
- **Parameter passing**: Parameters (min_val, max_val) are bitcast from float to uint32_t (IEEE 754 representation) and baked into the SFPU_OP_CHAIN compile-time define string as hex literals. This follows the relu_max pattern (e.g., `relu_max_tile(0, 0x40c00000u)`) and avoids needing runtime scalar arg packing.
- **Two-parameter registration**: Since no existing macro handles two float parameters with defaults, a custom inline function was defined in unary.hpp. A nanobind wrapper (`unary_two_float_5param_to_6param_wrapper`) bridges the 6-param C++ function to the 5-param nanobind binding (dropping sub_core_grids).
- **Old unary path only**: Registered in the old `unary_op_utils.cpp` path (not `unary_ng`) because the ng path's `get_op_init_and_func` doesn't accept parameters and can't embed them in the SFPU_OP_CHAIN string.

## Known Limitations
- No INT32 support (hardtanh is float-only).
- Parameters are baked as compile-time defines, so each unique (min_val, max_val) pair triggers a kernel recompile.
- Not registered in the unary_ng dispatch path.

## Test Results
- **Status**: PASS (after 1 test run, 2 implementation fixes)
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py
- **bfloat16** (is_fp32=False):
  - **Parameters tested**: default(-1,1), narrow(-0.5,0.5), relu6-like(0,6), wide(-2,2)
  - **Max ULP**: 0 (hardtanh is a clamp — exact on bfloat16)
  - **allclose**: PASS (rtol=1.6e-2, atol=1e-2)
- **fp32** (is_fp32=True):
  - **Parameters tested**: default(-1,1), narrow(-0.5,0.5), relu6-like(0,6), wide(-2,2)
  - **Max ULP**: 0 (hardtanh is a clamp — exact on fp32)
  - **allclose**: PASS (rtol=1e-3, atol=1e-4)

## Debug Log
### Fix 1: SFPU kernel signature mismatch
- **Error type**: build_error (on-device JIT compilation)
- **Error**: `too few arguments to function` at `llk_math_eltwise_unary_sfpu_params.h:31`
- **Hypothesis**: `calculate_hardtanh` takes `(iterations, param0, param1)` but `_llk_math_eltwise_unary_sfpu_params_` template calls `sfpu_func(args...)` where `args` is only `(param0, param1)`.
- **Fix**: Removed `iterations` parameter from `calculate_hardtanh`, changed inner loop from `for(d < iterations)` to `for(d < ITERATIONS)` using the template parameter.
- **Files modified**: ckernel_sfpu_hardtanh.h (wormhole_b0 and blackhole)

### Fix 2: fmt::format doc string escaping
- **Error type**: build_error (host compilation)
- **Error**: `argument not found` in fmt::format for hardtanh nanobind doc string
- **Hypothesis**: `{min_val}` and `{max_val}` interpreted as named format args by fmt::format.
- **Fix**: Escaped as `{{min_val}}` and `{{max_val}}` in the doc R"doc()" string.
- **Files modified**: unary_nanobind.cpp

### New Files
tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py
