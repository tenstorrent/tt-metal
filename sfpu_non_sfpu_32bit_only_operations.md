# SFPU Operations: Non-SFPU & 32-bit Only (Strict)

This document lists SFPU operations that:
1. **Do not utilize SFPU floating-point hardware** - They use integer ALU instructions instead of the SFPU's floating-point math unit
2. **Are implemented ONLY for 32-bit datatypes** - No 16-bit (LO16, FP16B, UINT16) support

---

## Operations

### 1. max_int32
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_max_int32.h`
- **Function:** `_calculate_max_int32_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses `TTI_SFPIADD` for integer comparison with condition codes (`TTI_SFPENCC`)
- **Datatype:** INT32 only

### 2. fill_int
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`
- **Function:** `_calculate_fill_int_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Simple data store operation - loads immediate value and stores to destination. No computation.
- **Datatype:** INT32 only (InstrModLoadStore::INT32)

### 3. zero_comp_int
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
- **Function:** `_calculate_zero_comp_int_<APPROXIMATION_MODE, COMP_MODE, ITERATIONS>`
- **Supported modes:** equal_zero, not_equal_zero, less_than_zero, greater_than_zero, less_than_equal_zero, greater_than_equal_zero
- **Why not SFPU:** Uses SFPI vInt operations with conditional branches - integer comparison logic
- **Datatype:** INT32 only (via sfpi::vInt)

### 4. comp_unary_int
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
- **Function:** `_calculate_comp_unary_int_<APPROXIMATION_MODE, COMP_MODE, ITERATIONS>`
- **Supported modes:** unary_eq, unary_ne, unary_gt, unary_lt, unary_ge, unary_le
- **Why not SFPU:** Complex integer comparison with sign handling using SFPI vInt conditional operations
- **Datatype:** INT32 only (via sfpi::vInt)

### 5. negative_int
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_negative.h`
- **Function:** `_calculate_negative_int_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses sign-magnitude format manipulation via reinterpret casts - not floating-point negation
- **Datatype:** INT32 only (sign-magnitude format)

### 6. typecast_fp32_to_int32
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`
- **Function:** `_calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses `TTI_SFPEXEXP`, `TTI_SFPEXMAN`, `TTI_SFPSHFT`, `TTI_SFPIADD` for extracting exponent/mantissa and constructing int32 result
- **Datatype:** FP32 → INT32 (32-bit to 32-bit)

### 7. typecast_fp32_to_uint32
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`
- **Function:** `_calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses bit manipulation for unsigned conversion
- **Datatype:** FP32 → UINT32 (32-bit to 32-bit)

### 8. typecast_int32_to_fp32
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`
- **Function:** `_calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses `TTI_SFPABS`, `TTI_SFPSHFT2`, `TTI_SFPCAST`, `TTI_SFPSETSGN` for sign handling and format conversion
- **Datatype:** INT32 → FP32 (32-bit to 32-bit)

### 9. typecast_uint32_to_fp32
- **Path:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`
- **Function:** `_calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>`
- **Why not SFPU:** Uses SFPLOADMACRO scheduling with `TTI_SFPSETSGN`, `TTI_SFPCAST`, `TTI_SFPMAD` for handling unsigned 32-bit conversion
- **Datatype:** UINT32 → FP32 (32-bit to 32-bit)

---

## Summary Table

| Operation | File | Datatype |
|-----------|------|----------|
| max_int32 | ckernel_sfpu_max_int32.h | INT32 |
| fill_int | ckernel_sfpu_fill.h | INT32 |
| zero_comp_int | ckernel_sfpu_comp.h | INT32 |
| comp_unary_int | ckernel_sfpu_comp.h | INT32 |
| negative_int | ckernel_sfpu_negative.h | INT32 |
| typecast_fp32_to_int32 | ckernel_sfpu_typecast.h | FP32 → INT32 |
| typecast_fp32_to_uint32 | ckernel_sfpu_typecast.h | FP32 → UINT32 |
| typecast_int32_to_fp32 | ckernel_sfpu_typecast.h | INT32 → FP32 |
| typecast_uint32_to_fp32 | ckernel_sfpu_typecast.h | UINT32 → FP32 |

---

## Notes

1. **All paths are relative to the repository root** and shown for Wormhole B0. Equivalent implementations exist in:
   - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/`

2. **Excluded operations** that have 16-bit support:
   - `add_int` (supports LO16)
   - `sub_int` (supports LO16)
   - `mul_int` (LO16 only)
   - `binary_bitwise` (configurable, supports LO16)
   - `shift` operations (support LO16)
   - typecast operations involving FP16B, UINT16
