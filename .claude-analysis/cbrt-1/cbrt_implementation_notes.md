# cbrt Implementation Notes

## Math Definition
x^(1/3) - Cube root function

## Algorithm
Uses the Moroz et al. magic constant method for fast cube root computation.
The algorithm uses bit manipulation (reinterpret cast to int, multiply by -1/3, shift)
to get an initial approximation, then refines with polynomial evaluation.

For fp32 mode, an additional Newton-Raphson refinement step is applied.
For bfloat16 mode, the result is truncated via float_to_fp16b.

## Implementation Summary

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` -- SFPU kernel with calculate_cube_root and cube_root_init
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` -- LLK dispatch (init + compute)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` -- Same SFPU kernel for Blackhole
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` -- Same LLK dispatch for Blackhole
- `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` -- Compute API (cbrt_tile, cbrt_tile_init)
- `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` -- Restored shared infrastructure (init_sfpu, unary_op_init_common)

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added `cbrt` to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added `cbrt` to SfpuType enum
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h` -- Added cbrt include, removed missing nuked includes
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h` -- Added cbrt include, removed missing nuked includes
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_CBRT_INCLUDE guard
- `tt_metal/hw/sources.cmake` -- Fixed to list only existing eltwise_unary files (removed nuked references)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Added CBRT to get_macro_definition and get_op_init_and_func_default
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- Added CBRT to get_macro_definition
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.hpp` -- Added DECLARE_UNARY_NG_OP(cbrt)
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.cpp` -- Added DEFINE_UNARY_NG_OP(cbrt, CBRT)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added bind_unary_operation_subcoregrids for cbrt
- `ttnn/ttnn/operations/unary.py` -- Added cbrt to golden function map and TTNN_ELTWISE_UNARY_CPP_FUNCTIONS

## Reference Operations Used
Most useful reference: the pre-nuke cbrt code itself (recovered from git history before the batch nuke commit db3f683e0a5).

## Key Design Decisions
1. cbrt is registered via the unary_ng path (DECLARE_UNARY_NG_OP / DEFINE_UNARY_NG_OP), NOT via the old REGISTER_UNARY_OPERATION macro in unary.hpp. This matches the pre-nuke pattern.
2. The golden function uses `torch.sgn(x) * torch.pow(torch.abs(x), 1.0/3)` to handle negative inputs correctly (Python's ** operator doesn't handle negative bases with fractional exponents).
3. The SFPU kernel uses 3 programmable constants (vConstFloatPrgm0/1/2) for polynomial coefficients in the refinement step.

## Known Issues
1. The batch nuke left broken references in multiple files (llk_math_unary_sfpu_api.h included headers for nuked operations). These were cleaned up as part of the cbrt implementation.
2. sources.cmake still referenced many deleted header files from the nuke. Fixed to only list existing files.
3. eltwise_unary.h was deleted by the nuke but is required by eltwise_sfpu.cpp. Restored as shared infrastructure.
4. Pre-existing build failures from the nuke (reciprocal, eqz, neg, cos, sin, etc.) prevent a full build. These are NOT related to cbrt.

## Build Status
cbrt code compiles cleanly. Full build fails due to pre-existing nuke damage in complex_binary, complex_unary, creation, and unary_nanobind modules that reference other nuked operations.
