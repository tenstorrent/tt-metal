# Analysis: softsign (reference for frac)

## SFPU Kernel Pattern
- **File**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
- **Namespace**: `ckernel::sfpu`
- **Template**: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- **Function**: `calculate_softsign()`
- **Init function**: `softsign_init()` (calls `_init_sfpu_reciprocal_`)
- **Includes**: `ckernel.h`, `ckernel_defs.h`, `sfpu/ckernel_sfpu_recip.h`

## Key SFPI Patterns
- Loop: `for (int d = 0; d < ITERATIONS; d++)` with `#pragma GCC unroll 8`
- Read: `sfpi::vFloat v = sfpi::dst_reg[0];`
- Write: `sfpi::dst_reg[0] = result;`
- Advance: `sfpi::dst_reg++;`
- Uses `sfpi::abs()`, `sfpi::vConst1`, basic arithmetic

## LLK Dispatch Pattern
- **File**: `llk_math_eltwise_unary_sfpu_softsign.h`
- Init: `llk_math_eltwise_unary_sfpu_init<SfpuType::softsign, APPROXIMATE>(ckernel::sfpu::softsign_init<APPROXIMATE>)`
- Compute: `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`

## Compute API Pattern
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`
- `#include "api/compute/common_globals.h"` + `#ifdef TRISC_MATH` guard
- `softsign_tile(uint32_t idst)` calls `MATH((llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)))`
- `softsign_tile_init()` calls `MATH((llk_math_eltwise_unary_sfpu_softsign_init<APPROX>()))`

## Registration Pattern (unary_op_utils.cpp)
- `get_macro_definition`: `case UnaryOpType::SOFTSIGN: return "SFPU_OP_SOFTSIGN_INCLUDE";`
- `get_op_init_and_func_default`: `case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};`
- `sfpu_split_includes.h`: `#if SFPU_OP_SOFTSIGN_INCLUDE` -> `#include "api/compute/eltwise_unary/softsign.h"`

## unary_ng Registration
- In `unary_ng_op_utils.cpp`: `case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};`

## SfpuType Enum
- `SfpuType::softsign` in `llk_sfpu_types.h`

## Python/TTNN Binding
- `unary.hpp`: `REGISTER_UNARY_OPERATION(softsign, SOFTSIGN)`
- `unary_nanobind.cpp`: `bind_unary_operation<"softsign", &ttnn::softsign>(...)`
- Golden: `ttnn.attach_golden_function(ttnn.softsign, _softsign_golden_function)` in experimental_loader
