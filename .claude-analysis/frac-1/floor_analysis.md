# floor — SFPU Kernel Analysis

## Overview
- **Operation**: floor
- **Math**: floor(x) — largest integer <= x
- **UnaryOpType**: FLOOR

## Layer Architecture

### 1. SFPU Kernel (ckernel_sfpu_rounding_ops.h in LLK)
- Located in `tt_metal/third_party/tt_llk/.../common/inc/sfpu/`
- `_floor_body_()`: calls `_trunc_body_()`, then adjusts for negative values
- `_calculate_floor_()`: iteration loop, SFPLOAD -> _floor_body_ -> SFPSTORE

### 2. LLK Dispatch
- Part of rounding ops family in LLK submodule

### 3. Compute API (rounding.h)
- Located in `tt_metal/hw/inc/api/compute/eltwise_unary/rounding.h`
- `floor_tile(uint32_t idst)` — `MATH(SFPU_TWO_PARAM_KERNEL(_calculate_floor_, APPROX, 8, idst, (int)VectorMode::RC))`
- `rounding_op_tile_init()` — `MATH(SFPU_UNARY_KERNEL_INIT(unused, APPROX))`
- Shared init function with ceil, trunc, frac, round, stochastic_round

### 4. No Split Include
- Uses default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`
- The `rounding.h` header is included as part of standard compute kernel API

### 5. Op Utils — NOT registered in old `unary_op_utils.cpp`
- FLOOR is NOT in `get_op_init_and_func_default()` in the worktree code
- However, it IS registered in the `unary_ng` path

### 6. Unary_ng (unary_ng_op_utils.cpp)
- `case UnaryOpType::FLOOR: return {"rounding_op_tile_init();", fmt::format("floor_tile({});", idst)};`

### 7. Python Binding
- `REGISTER_UNARY_OPERATION(floor, FLOOR)` in `unary.hpp`
- Auto-bound to Python as `ttnn.floor`

### 8. No standalone golden function
- `ttnn.floor` maps directly via decorator system

## Key Pattern Notes
- Same init function for all rounding ops: `rounding_op_tile_init()`
- No custom include macro needed
- Uses `SFPU_TWO_PARAM_KERNEL` instead of `SFPU_UNARY_KERNEL`
- Not in old `unary_op_utils.cpp` but IS in `unary_ng_op_utils.cpp`
