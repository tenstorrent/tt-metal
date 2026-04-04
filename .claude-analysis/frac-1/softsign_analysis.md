# softsign — SFPU Kernel Analysis

## Overview
- **Operation**: softsign
- **Math**: x / (1 + |x|)
- **UnaryOpType**: SOFTSIGN

## Layer Architecture

### 1. SFPU Kernel (ckernel_sfpu_softsign.h)
- Located in `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/`
- Contains `_calculate_softsign_()` template function
- Uses SFPI instructions directly (TTI_ macros)

### 2. LLK Dispatch (llk_math_eltwise_unary_sfpu_softsign.h)
- Located alongside ckernel in `llk_sfpu/`
- Provides `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)` and `_init()` functions
- Uses SFPU_UNARY_KERNEL macros

### 3. Compute API (softsign.h)
- Located in `tt_metal/hw/inc/api/compute/eltwise_unary/`
- Defines `softsign_tile(uint32_t idst)` and `softsign_tile_init()`
- Uses `MATH(llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst))`

### 4. Split Includes (sfpu_split_includes.h)
- Guarded by `#if SFPU_OP_SOFTSIGN_INCLUDE`
- Includes `softsign.h` only when this macro is defined

### 5. Macro Definition (unary_op_utils.cpp)
- `get_macro_definition()` returns `"SFPU_OP_SOFTSIGN_INCLUDE"` for `SOFTSIGN`

### 6. Op Utils (unary_op_utils.cpp)
- `get_op_init_and_func_default()` returns:
  - init: `"softsign_tile_init();"`
  - func: `fmt::format("softsign_tile({});", idst)`

### 7. Unary_ng (unary_ng_op_utils.cpp)
- Also registered in the new-gen system

### 8. Python Nanobind (unary_nanobind.cpp)
- Bound via `bind_unary_operation<"softsign", &ttnn::softsign>(...)`

### 9. Golden Function (unary.py)
- In `TTNN_ELTWISE_UNARY_CPP_FUNCTIONS` list
- Maps to `torch.nn.functional.softsign`

## Key Pattern Notes
- Uses custom include macro (`SFPU_OP_SOFTSIGN_INCLUDE`) for split includes
- Parameterless operation (no extra float/int params)
- Registered in both old and new op utils
