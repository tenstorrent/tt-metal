# silu — SFPU Kernel Analysis

## Overview
- **Operation**: silu (Sigmoid Linear Unit)
- **Math**: x * sigmoid(x) = x / (1 + exp(-x))
- **UnaryOpType**: SILU

## Layer Architecture

### 1. SFPU Kernel
- Part of the LLK submodule (not custom ckernel)
- Uses standard sigmoid and multiply primitives

### 2. LLK Dispatch
- `llk_math_eltwise_unary_sfpu_silu` (in LLK submodule)

### 3. Compute API
- `silu_tile(uint32_t idst)` and `silu_tile_init()`
- Part of the standard compute kernel API

### 4. No Split Include
- Uses default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (no custom include needed)

### 5. Op Utils (unary_op_utils.cpp)
- `get_op_init_and_func_default()` returns:
  - init: `"silu_tile_init();"`
  - func: `fmt::format("silu_tile({});", idst)`

### 6. Unary_ng (unary_ng_op_utils.cpp)
- Registered: `case UnaryOpType::SILU: return {"silu_tile_init();", fmt::format("silu_tile({});", idst)};`

### 7. Python Nanobind (unary_nanobind.cpp)
- Bound via `bind_unary_operation<"silu", &ttnn::silu>(...)`

### 8. Golden Function (unary.py)
- In `TTNN_ELTWISE_UNARY_CPP_FUNCTIONS` list
- Maps to `torch.nn.functional.silu`

### 9. String-to-Op (unary_op_utils.cpp)
- Registered in `string_to_unary_with_param()`: `if (name == "silu") return UnaryWithParam(UnaryOpType::SILU);`

## Key Pattern Notes
- Uses default include macro (standard LLK includes suffice)
- Parameterless operation
- Full integration in old, new, nanobind, and golden paths
