# ceil — SFPU Kernel Analysis

## Overview
- **Operation**: ceil
- **Math**: ceil(x) — smallest integer >= x
- **UnaryOpType**: CEIL

## Layer Architecture
Identical pattern to floor (same family).

### 1. SFPU Kernel (ckernel_sfpu_rounding_ops.h in LLK)
- `_ceil_body_()`: calls `_trunc_body_()`, then adjusts for positive values
- `_calculate_ceil_()`: iteration loop

### 2. Compute API (rounding.h)
- `ceil_tile(uint32_t idst)` — `MATH(SFPU_TWO_PARAM_KERNEL(_calculate_ceil_, APPROX, 8, idst, (int)VectorMode::RC))`
- Shared `rounding_op_tile_init()`

### 3. Unary_ng
- `case UnaryOpType::CEIL: return {"rounding_op_tile_init();", fmt::format("ceil_tile({});", idst)};`

### 4. Not in old unary_op_utils.cpp
- Same as floor: missing from `get_op_init_and_func_default()`

## Key Pattern Notes
- Exact same pattern as floor, trunc, frac
- All share `rounding_op_tile_init()`
- API header: `rounding.h`
