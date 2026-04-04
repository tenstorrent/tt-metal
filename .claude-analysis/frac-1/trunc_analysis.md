# trunc — SFPU Kernel Analysis

## Overview
- **Operation**: trunc
- **Math**: trunc(x) — rounds towards zero (removes fractional part, keeps sign)
- **UnaryOpType**: TRUNC

## Layer Architecture

### 1. SFPU Kernel (ckernel_sfpu_rounding_ops.h in LLK)
- `_trunc_body_()`: Core implementation used by ALL rounding ops
  - Sets L3=23
  - Creates bit mask 0x8000_0000
  - Disables lanes where exp < 0
  - Computes exp = 23 - exp
  - Shifts mask: mask <<= exp
  - Applies mask to clear fractional bits
- `_calculate_trunc_()`: iteration loop, SFPLOAD -> _trunc_body_ -> SFPSTORE

### 2. Compute API (rounding.h)
- `trunc_tile(uint32_t idst)` — `MATH(SFPU_TWO_PARAM_KERNEL(_calculate_trunc_, APPROX, 8, idst, (int)VectorMode::RC))`
- Shared `rounding_op_tile_init()`

### 3. Unary_ng
- `case UnaryOpType::TRUNC: return {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};`

### 4. Relationship to frac
- `frac` is implemented as: `frac(x) = x - trunc(x)` in the SFPU kernel
- `_calculate_frac_()` calls `_trunc_body_()` then subtracts:
  ```
  TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG1, 0);
  ```
  This computes: L1 = L0 + L1 * (-1) = x + trunc(x) * (-1) = x - trunc(x)

## Key Pattern Notes
- Foundation operation for the entire rounding family
- frac directly depends on trunc at the SFPU kernel level
- All rounding ops share the same init: `rounding_op_tile_init()`
