# CBRT Reference Analysis

## Pre-Nuke Architecture (Recovered from Git)

The cbrt operation was implemented across the following layers:

### Layer 1: SFPU Kernel (`ckernel_sfpu_cbrt.h`)
- **Location**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
- **Algorithm**: Modified Moroz et al. magic constant method for fast cube root
- **Template params**: `APPROXIMATION_MODE`, `is_fp32_dest_acc_en`, `ITERATIONS`
- **Init function**: `cube_root_init<APPROXIMATE>()` - sets 3 programmable constants
- **Compute function**: `calculate_cube_root<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>()`
- **Key detail**: Uses `vConstFloatPrgm0/1/2` for polynomial coefficients
- **Key detail**: fp32 path has an extra Newton refinement step
- **Key detail**: bfloat16 path uses `float_to_fp16b()` to truncate result

### Layer 2: LLK Dispatch (`llk_math_eltwise_unary_sfpu_cbrt.h`)
- **Location**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
- **Init**: `llk_math_eltwise_unary_sfpu_cbrt_init<APPROXIMATE>()` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`
- **Compute**: `llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>(dst_index, vector_mode)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<...>, dst_index, vector_mode)`

### Layer 3: Compute API (`cbrt.h`)
- **Location**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
- **Functions**: `cbrt_tile(uint32_t idst)` and `cbrt_tile_init()`
- **Pattern**: `ALWI void cbrt_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst))); }`

### Layer 4: Split Includes (`sfpu_split_includes.h`)
- **Macro**: `SFPU_OP_CBRT_INCLUDE`
- **Pattern**: `#if SFPU_OP_CBRT_INCLUDE` / `#include "api/compute/eltwise_unary/cbrt.h"` / `#endif`

### Layer 5: SfpuType Enum
- **Entry**: `cbrt` in `SfpuType` enum in `llk_sfpu_types.h`

### Layer 6: UnaryOpType Enum
- **Entry**: `CBRT` (already preserved - was NOT nuked)

### Layer 7: LLK API Include (`llk_math_unary_sfpu_api.h`)
- **Include**: `#include "llk_math_eltwise_unary_sfpu_cbrt.h"`

### Layer 8: Op Utils (`unary_op_utils.cpp`)
- **get_macro_definition**: `case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";`
- **get_op_init_and_func_default**: `case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};`
- **get_op_approx_mode**: No special entry (returns false by default)

### Layer 9: unary_ng Utils (already present)
- **Line 121**: `case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};`

### Layer 10: C++ API Registration (`unary.hpp` / `unary_ng.hpp`)
- **unary.hpp**: Used `REGISTER_UNARY_OPERATION(cbrt, CBRT)` macro
- **unary_ng.hpp**: Used `DECLARE_UNARY_NG_OP(cbrt)` macro
- **unary_ng.cpp**: Used `DEFINE_UNARY_NG_OP(cbrt, CBRT)` macro

### Layer 11: Python Binding (`unary_nanobind.cpp`)
- **Binding**: `bind_unary_operation_subcoregrids<"cbrt">(mod, &ttnn::cbrt, R"doc(...)doc", "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");`

### Layer 12: Golden Function (`unary.py`)
- **torch_cbrt**: `torch.sgn(x) * torch.pow(torch.abs(x), 1.0/3)` (already present in file)
- **Registration**: Needs to be added to `name_to_golden_function` dict and `TTNN_ELTWISE_UNARY_CPP_FUNCTIONS` list

## Files to Create
1. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
2. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
3. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
4. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
5. `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`

## Files to Modify
1. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` - Add `cbrt` to SfpuType
2. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` - Add `cbrt` to SfpuType
3. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h` - Include cbrt LLK
4. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h` - Include cbrt LLK
5. `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` - Add CBRT include guard
6. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` - Register in get_macro_definition, get_op_init_and_func_default
7. `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` - Register with REGISTER_UNARY_OPERATION macro
8. `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.hpp` - Declare with DECLARE_UNARY_NG_OP
9. `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.cpp` - Define with DEFINE_UNARY_NG_OP
10. `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` - Bind cbrt
11. `ttnn/ttnn/operations/unary.py` - Register golden function
