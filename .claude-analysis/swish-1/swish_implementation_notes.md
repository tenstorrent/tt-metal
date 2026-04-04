# swish (SiLU) Implementation Notes

## Overview
- **Operation**: swish / silu
- **Math definition**: x * sigmoid(x) = x / (1 + exp(-x))
- **UnaryOpType enum entry**: SILU (pre-existing)
- **Date**: 2026-04-04

## Key Design Decision

Unlike other SFPU operations (selu, cosh, cbrt, hardsigmoid, hardtanh) that required creating new SFPU kernel files from scratch, the silu/swish operation already has a complete SFPU kernel implementation in the upstream `tt_llk` third-party library:

- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h`

The LLK dispatch headers are also present in the upstream library and are copied to the build output during the build process:
- `build_*/libexec/tt-metalium/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h`
- `build_*/libexec/tt-metalium/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`

Additionally, the compute API (`silu_tile` and `silu_tile_init`) and the `REGISTER_UNARY_OPERATION(silu, SILU)` macro and `UnaryOpType::SILU` enum entry were already present. The implementation therefore focused on the **software stack integration** layer that was missing.

## What Was Already Present (Pre-existing)
1. SFPU kernel in `tt_llk` submodule (ckernel_sfpu_silu.h)
2. LLK dispatch header in `tt_llk` submodule (llk_math_eltwise_unary_sfpu_silu.h)
3. Include in `llk_math_unary_sfpu_api.h` (both wormhole_b0 and blackhole)
4. Compute API functions `silu_tile()` and `silu_tile_init()` in `compute_kernel_api.h`
5. `REGISTER_UNARY_OPERATION(silu, SILU)` in `unary.hpp`
6. `UnaryOpType::SILU` in `unary_op_types.hpp`
7. `unary_ng_op_utils.cpp` case for `UnaryOpType::SILU`

## What Was Implemented (New Changes)

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added `silu` to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added `silu` to SfpuType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Added SILU case to `get_op_init_and_func_default` and `string_to_unary_with_param`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added nanobind binding for silu
- `ttnn/ttnn/operations/unary.py` -- Added silu golden function and registration

### New Files
- `tests/ttnn/unit_tests/operations/eltwise/test_silu.py` -- Test file

## Reference Operations Used
- **selu**: Pattern for `get_op_init_and_func_default`, `SfpuType` enum, golden function registration
- **hardsigmoid**: Pattern for nanobind binding, `unary_op_utils.cpp` registration
- **cosh**: Pattern for golden function registration (standalone function style)

## Technical Notes

1. **No split includes needed**: silu uses the standard `compute_kernel_api.h` path (always included), not the conditional split-includes mechanism. Therefore, no `SFPU_OP_SILU_INCLUDE` macro or entry in `sfpu_split_includes.h` is needed.

2. **Approx mode**: silu uses `get_op_approx_mode` default (false). The upstream kernel handles approximation internally.

3. **DST_ACCUM_MODE**: The `silu_tile` function in `compute_kernel_api.h` passes `DST_ACCUM_MODE` as a template parameter to the LLK function, which differs from the split-include ops that use `SFPU_UNARY_NO_PARAM_KERNEL_FN`.

## Known Limitations
- None. The operation is a straightforward integration of an upstream kernel with no parameters.
