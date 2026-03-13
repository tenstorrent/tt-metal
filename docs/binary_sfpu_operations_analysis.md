# Binary SFPU Operations Analysis

This document catalogs all binary SFPU operations in the TTNN framework, grouped by type and complexity. It covers the `SfpuBinaryOp` enum (28 operations) and the composite `BinaryOpType` entries that are decomposed into FPU + unary pre/post-processing steps rather than dispatching to a native SFPU kernel.

## Source Files

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp` | `BinaryOpType` enum (43 entries) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` | `SfpuBinaryOp` enum (28 entries), `FpuBinaryOp` enum (3 entries) |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` | `OpConfig` constructor (dispatch logic), `get_sfpu_init_fn` (kernel mapping) |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` | `is_binary_sfpu_op()` (legacy path routing) |
| `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` | Core FP binary SFPU tile functions |
| `tt_metal/hw/inc/api/compute/*.h` | Specialized SFPU tile functions (bitwise, shift, int, etc.) |

---

## FPU vs SFPU: When Does Each Execute?

The Tensix core has two compute engines:

- **FPU (Matrix Engine)**: Handles `ADD`, `SUB`, `MUL` natively for BFloat16 via the matrix unit. Very fast, but limited to these three operations.
- **SFPU (Vector Engine)**: Handles all other element-wise operations via the vector unit. Also handles `ADD`, `SUB`, `MUL` when the data type is Float32, Int32, UInt32, or UInt16 (types that require 32-bit precision the FPU path doesn't support).

The routing decision is made by `is_binary_sfpu_op()` and the `OpConfig` constructor. Key rule: **if the data type is BFloat16 and the operation is ADD/SUB/MUL, the FPU path is used; otherwise, the SFPU path is used.**

---

## Group 1: Basic Arithmetic (Native SFPU)

These are direct single-instruction SFPU operations — one `llk_math` call per tile.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `ADD` | `y = a + b` | `add_binary_tile` / `add_int_tile<Fmt>` | `eltwise_binary_sfpu.h` / `add_int_sfpu.h` | FP32, Int32, UInt32, UInt16 | **EASY** | FP path uses `BinaryOp::ADD` template; Int path uses dedicated int kernel |
| `SUB` | `y = a - b` | `sub_binary_tile` / `sub_int_tile<Fmt>` | `eltwise_binary_sfpu.h` / `sub_int_sfpu.h` | FP32, Int32, UInt32, UInt16 | **EASY** | Same split as ADD |
| `MUL` | `y = a * b` | `mul_binary_tile` / `mul_int_tile<Fmt>` | `eltwise_binary_sfpu.h` / `mul_int_sfpu.h` | FP32, Int32, UInt32, UInt16 | **EASY** | Uses `DST_ACCUM_MODE` template param |
| `RSUB` | `y = b - a` | `rsub_binary_tile` / `rsub_int_tile<Fmt>` | `eltwise_binary_sfpu.h` / `sub_int_sfpu.h` | FP32, Int32, UInt32, UInt16 | **EASY** | Reverse subtraction |

**Complexity**: **EASY**. Single SFPU instruction per element. These are the simplest possible SFPU binary ops.

---

## Group 2: Division and Modular Arithmetic

These operations involve division, which is inherently more expensive on the SFPU due to iterative reciprocal computation (for FP) or multi-step integer division algorithms.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `DIV` | `y = a / b` | `div_binary_tile` / `div_int32_tile` | `eltwise_binary_sfpu.h` / `div_int32_sfpu.h` | FP32 (SFPU div), Int32 (int div) | **MEDIUM** | FP: iterative reciprocal + mul; Int: multi-step |
| `DIV_FLOOR` | `y = floor(a / b)` | `div_int32_floor_tile` | `div_int32_floor.h` | Int32 only | **MEDIUM** | Floor division (Python-style); division + floor |
| `DIV_TRUNC` | `y = trunc(a / b)` | `div_int32_trunc_tile` | `div_int32_floor.h` | Int32 only | **MEDIUM** | Truncated division (C-style); division + truncation |
| `REMAINDER` | `y = a - floor(a/b)*b` | `remainder_int32_tile` | `remainder_int32.h` | Int32 only | **MEDIUM** | Python-style remainder; division + multiply + subtract |
| `FMOD` | `y = a - trunc(a/b)*b` | `fmod_int32_tile` / `fmod_binary_tile` | `binary_fmod.h` | Int32, FP (BF16/FP32) | **MEDIUM** | C-style fmod; division + multiply + subtract; FP uses `DST_ACCUM_MODE` |

**Complexity**: **MEDIUM**. Division requires multiple SFPU cycles. Integer division is particularly expensive as it involves iterative subtraction or shift-based algorithms on the SFPU. FMOD and REMAINDER layer additional steps on top of division.

---

## Group 3: Bitwise Operations (Integer-Only)

These operate on raw bit patterns. SFPU-exclusive — no FPU path exists.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `BITWISE_AND` | `y = a & b` | `bitwise_and_binary_tile<Fmt>` | `binary_bitwise_sfpu.h` | Int32, UInt32, UInt16 | **EASY** | Templated on `DataFormat` |
| `BITWISE_OR` | `y = a \| b` | `bitwise_or_binary_tile<Fmt>` | `binary_bitwise_sfpu.h` | Int32, UInt32, UInt16 | **EASY** | |
| `BITWISE_XOR` | `y = a ^ b` | `bitwise_xor_binary_tile<Fmt>` | `binary_bitwise_sfpu.h` | Int32, UInt32, UInt16 | **EASY** | |

**Complexity**: **EASY**. Single SFPU instruction per element. These map directly to bitwise hardware operations.

---

## Group 4: Shift Operations (Integer-Only)

Bit shift operations. SFPU-exclusive — no FPU path exists.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `LEFT_SHIFT` | `y = a << b` | `binary_left_shift_tile<Fmt>` | `binary_shift.h` | Int32, UInt32 | **EASY** | Templated on `DataFormat` |
| `RIGHT_SHIFT` | `y = a >> b` (arithmetic) | `binary_right_shift_tile<Fmt>` | `binary_shift.h` | Int32, UInt32 | **EASY** | Sign-extending for signed types |
| `LOGICAL_RIGHT_SHIFT` | `y = a >>> b` (logical) | `binary_logical_right_shift_tile<Fmt>` | `binary_shift.h` | Int32, UInt32 | **EASY** | Zero-filling regardless of sign |

**Complexity**: **EASY**. Single SFPU instruction per element. Direct hardware shift operations.

---

## Group 5: Comparison Operations (Integer SFPU Path)

For **Int32** data types, comparisons run as native SFPU operations. For **BFloat16/FP32**, they are decomposed as `SUB` (FPU) + unary postprocess (e.g., `LTZ`, `GTZ`).

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `LT` | `y = (a < b) ? 1 : 0` | `lt_int32_tile` | `binary_comp.h` | Int32 only | **EASY** | FP types use `SUB` + `LTZ` unary |
| `GT` | `y = (a > b) ? 1 : 0` | `gt_int32_tile` | `binary_comp.h` | Int32 only | **EASY** | FP types use `SUB` + `GTZ` unary |
| `GE` | `y = (a >= b) ? 1 : 0` | `ge_int32_tile` | `binary_comp.h` | Int32 only | **EASY** | FP types use `SUB` + `GEZ` unary |
| `LE` | `y = (a <= b) ? 1 : 0` | `le_int32_tile` | `binary_comp.h` | Int32 only | **EASY** | FP types use `SUB` + `LEZ` unary |
| `EQ` | `y = (a == b) ? 1 : 0` | `eq_binary_tile` | `eltwise_binary_sfpu.h` | FP32 only | **EASY** | BFloat16 uses `SUB` + `EQZ` unary |

**Complexity**: **EASY** (native path). Each is a single SFPU comparison instruction. However, EQ only takes the native SFPU path for FP32; BFloat16 and Int32 use the decomposed path.

---

## Group 6: Min/Max Operations

Element-wise minimum and maximum with type-specialized variants.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `MAXIMUM` | `y = max(a, b)` | `binary_max_tile` / `binary_max_int32_tile` / `binary_max_uint32_tile` | `binary_max_min.h` | Any (BF16/FP32/Int32/UInt32) | **EASY** | FP variant supports `VectorMode` param |
| `MINIMUM` | `y = min(a, b)` | `binary_min_tile` / `binary_min_int32_tile` / `binary_min_uint32_tile` | `binary_max_min.h` | Any (BF16/FP32/Int32/UInt32) | **EASY** | FP variant supports `VectorMode` param |

**Complexity**: **EASY**. Single SFPU comparison + conditional select per element. Three separate kernel implementations cover FP, Int32, and UInt32.

---

## Group 7: Transcendental / Mathematical Operations

Operations involving logarithms, exponentiation, or multi-step mathematical computation.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `POWER` | `y = a^b` | `power_binary_tile` | `eltwise_binary_sfpu.h` | Any | **HARD** | Uses `DST_ACCUM_MODE`; internally computes `exp(b * log(a))` |
| `XLOGY` | `y = a * log(b)` (0 if a==0) | `xlogy_binary_tile` | `xlogy.h` | Any | **HARD** | Compound: multiply + log + conditional |

**Complexity**: **HARD**. `POWER` requires log + multiply + exp — three transcendental operations. `XLOGY` requires log + multiply + zero-check. These are the most computationally expensive native SFPU binary ops.

---

## Group 8: Number Theory (Integer-Only)

Iterative algorithms implementing GCD and LCM.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `GCD` | `y = gcd(a, b)` | `gcd_tile` | `gcd.h` | Int32, UInt32 | **HARD** | Euclidean algorithm; iterative |
| `LCM` | `y = lcm(a, b)` | `lcm_tile` | `lcm.h` | Int32, UInt32 | **HARD** | `lcm = |a*b| / gcd(a,b)`; limited to \|value\| <= 32767 |

**Complexity**: **HARD**. GCD uses an iterative Euclidean algorithm with unpredictable iteration count. LCM builds on GCD and adds multiplication + division. The iteration count depends on input values, making these variable-latency operations.

---

## Group 9: Quantization Operations

Specialized for integer quantization workflows. Require a `zero_point` runtime argument passed at init time.

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `QUANT` | `y = round(a * b) + zp` | `quant_tile` | `quantization.h` | Int32 | **MEDIUM** | Per-tensor affine quantization; `b` is scale |
| `REQUANT` | `y = round(a * b) + zp` | `requant_tile` | `quantization.h` | Int32 | **MEDIUM** | Re-quantization (int-to-int scale change) |
| `DEQUANT` | `y = (a - zp) * b` | `dequant_tile` | `quantization.h` | Int32 | **MEDIUM** | Inverse quantization; `b` is scale |

**Complexity**: **MEDIUM**. Each involves multiply + round/add with a runtime-configured zero point loaded into SFPU registers at init. Not iterative, but multi-step.

---

## Group 10: Conditional / Ternary

| SfpuBinaryOp | Math | Kernel Function | API Header | Data Types | Complexity | Notes |
|---|---|---|---|---|---|---|
| `WHERE` | `y = cond ? a : b` | `where_tile<Fmt>` | `eltwise_unary/where.h` | FP32, Int32, UInt32, BF16 | **MEDIUM** | Ternary-like but mapped as binary; condition comes from a third CB; used by `WHERE_TST` and `WHERE_TTS` BinaryOpTypes |

**Complexity**: **MEDIUM**. Element-wise conditional select with three inputs (condition + two values). Uses a dedicated SFPU path with `DataFormat` template specialization.

---

## Group 11: Composite Operations (FPU + Unary Pre/Post-Processing)

These `BinaryOpType` values do **not** map to a native `SfpuBinaryOp`. Instead, they are decomposed by the `OpConfig` constructor into:
1. Optional unary **pre-processing** on input A and/or B
2. A simple **FPU binary op** (ADD, SUB, or MUL)
3. Optional unary **post-processing** on the result

These run on the FPU (for BFloat16) or SFPU (for FP32/Int32), but the binary operation itself is always one of ADD/SUB/MUL.

| BinaryOpType | Decomposition | Pre-A | Pre-B | Binary | Post | Complexity | Notes |
|---|---|---|---|---|---|---|---|
| `NE` | `neq(a-b, 0)` | — | — | SUB | `NEZ` | **EASY** | Not-equal via subtraction + non-zero check |
| `SQUARED_DIFFERENCE` | `(a-b)^2` | — | — | SUB | `SQUARE` | **EASY** | Single post-process |
| `LT` (FP) | `ltz(a-b)` | — | — | SUB | `LTZ` | **EASY** | FP comparison via subtract |
| `GT` (FP) | `gtz(a-b)` | — | — | SUB | `GTZ` | **EASY** | |
| `GE` (FP) | `gez(a-b)` | — | — | SUB | `GEZ` | **EASY** | |
| `LE` (FP) | `lez(a-b)` | — | — | SUB | `LEZ` | **EASY** | |
| `EQ` (BF16) | `eqz(a-b)` | — | — | SUB | `EQZ` | **EASY** | |
| `RSUB` (FPU) | `(-a) + b` | `NEG` | — | ADD | — | **EASY** | FPU path only (BFloat16) |
| `ADDALPHA` | `a + alpha*b` | — | (scale) | ADD | — | **EASY** | Alpha-scaled addition (handled separately) |
| `SUBALPHA` | `a - alpha*b` | — | (scale) | SUB | — | **EASY** | Alpha-scaled subtraction (handled separately) |
| `LOGICAL_AND` | `nez(nez(a) * nez(b))` | `NEZ` | `NEZ` | MUL | `NEZ` | **MEDIUM** | Boolean AND via multiply; 3 unary steps |
| `LOGICAL_OR` | `nez(nez(a) + nez(b))` | `NEZ` | `NEZ` | ADD | `NEZ` | **MEDIUM** | Boolean OR via add; 3 unary steps |
| `LOGICAL_XOR` | `nez(nez(a) - nez(b))` | `NEZ` | `NEZ` | SUB | `NEZ` | **MEDIUM** | Boolean XOR via subtract; 3 unary steps |
| `DIV` (FPU) | `a * recip(b)` | — | `RECIP` | MUL | — | **MEDIUM** | FPU path only (BFloat16); reciprocal is multi-step |
| `BIAS_GELU` | `gelu(a+b)` | — | — | ADD | `GELU` | **MEDIUM** | GELU post-process is a multi-step approximation |
| `LDEXP` | `a * 2^b` | — | `EXP2` | MUL | — | **HARD** | Transcendental pre-process (EXP2) |
| `LOGADDEXP` | `log(exp(a) + exp(b))` | `EXP` | `EXP` | ADD | `LOG` | **HARD** | 3 transcendental operations |
| `LOGADDEXP2` | `log2(2^a + 2^b)` | `EXP2` | `EXP2` | ADD | `LOG2` | **HARD** | 3 transcendental operations |
| `HYPOT` | `sqrt(a^2 + b^2)` | `SQUARE` | `SQUARE` | ADD | `SQRT` | **HARD** | 2 pre-processes + sqrt post-process |

**Complexity**: Varies. The binary op itself is trivial (ADD/SUB/MUL), but total cost depends on the unary operations:
- **EASY**: `NE`, `SQUARED_DIFFERENCE`, comparisons, `RSUB`/`ADDALPHA`/`SUBALPHA` (one simple unary step or none)
- **MEDIUM**: `LOGICAL_AND/OR/XOR` (three cheap unary steps), `BIAS_GELU` (GELU approximation), `DIV` (reciprocal)
- **HARD**: `LOGADDEXP/LOGADDEXP2` (three transcendental ops), `HYPOT` (two pre + sqrt), `LDEXP` (transcendental pre)

---

## Complexity Summary — Operations Grouped by Difficulty

### EASY (19 native + 10 composite = 29 total)

Single SFPU instruction per element, or a simple FPU op with one cheap unary step. Minimal compute cost.

| # | Operation | Type | Math | Why EASY |
|---|---|---|---|---|
| 1 | `ADD` | Native SFPU | `a + b` | Single instruction |
| 2 | `SUB` | Native SFPU | `a - b` | Single instruction |
| 3 | `MUL` | Native SFPU | `a * b` | Single instruction |
| 4 | `RSUB` | Native SFPU | `b - a` | Single instruction (operand swap) |
| 5 | `BITWISE_AND` | Native SFPU | `a & b` | Single instruction, integer-only |
| 6 | `BITWISE_OR` | Native SFPU | `a \| b` | Single instruction, integer-only |
| 7 | `BITWISE_XOR` | Native SFPU | `a ^ b` | Single instruction, integer-only |
| 8 | `LEFT_SHIFT` | Native SFPU | `a << b` | Single instruction, integer-only |
| 9 | `RIGHT_SHIFT` | Native SFPU | `a >> b` | Single instruction, integer-only |
| 10 | `LOGICAL_RIGHT_SHIFT` | Native SFPU | `a >>> b` | Single instruction, integer-only |
| 11 | `LT` (Int32) | Native SFPU | `a < b ? 1 : 0` | Single comparison instruction |
| 12 | `GT` (Int32) | Native SFPU | `a > b ? 1 : 0` | Single comparison instruction |
| 13 | `GE` (Int32) | Native SFPU | `a >= b ? 1 : 0` | Single comparison instruction |
| 14 | `LE` (Int32) | Native SFPU | `a <= b ? 1 : 0` | Single comparison instruction |
| 15 | `EQ` (FP32) | Native SFPU | `a == b ? 1 : 0` | Single comparison instruction |
| 16 | `MAXIMUM` | Native SFPU | `max(a, b)` | Compare + conditional select |
| 17 | `MINIMUM` | Native SFPU | `min(a, b)` | Compare + conditional select |
| 18 | `NE` | Composite | `nez(a - b)` | SUB + one cheap unary post |
| 19 | `SQUARED_DIFFERENCE` | Composite | `(a - b)^2` | SUB + one cheap unary post |
| 20 | `LT` (FP) | Composite | `ltz(a - b)` | SUB + one cheap unary post |
| 21 | `GT` (FP) | Composite | `gtz(a - b)` | SUB + one cheap unary post |
| 22 | `GE` (FP) | Composite | `gez(a - b)` | SUB + one cheap unary post |
| 23 | `LE` (FP) | Composite | `lez(a - b)` | SUB + one cheap unary post |
| 24 | `EQ` (BF16) | Composite | `eqz(a - b)` | SUB + one cheap unary post |
| 25 | `RSUB` (FPU) | Composite | `(-a) + b` | NEG pre + ADD |
| 26 | `ADDALPHA` | Composite | `a + alpha*b` | Scaled ADD |
| 27 | `SUBALPHA` | Composite | `a - alpha*b` | Scaled SUB |

### MEDIUM (9 native + 5 composite = 14 total)

Multi-step but non-iterative. Division variants, quantization with runtime params, conditional select, or 2-3 cheap unary steps around a binary op.

| # | Operation | Type | Math | Why MEDIUM |
|---|---|---|---|---|
| 1 | `DIV` | Native SFPU | `a / b` | Iterative reciprocal (FP) or multi-step int division |
| 2 | `DIV_FLOOR` | Native SFPU | `floor(a / b)` | Integer division + floor rounding |
| 3 | `DIV_TRUNC` | Native SFPU | `trunc(a / b)` | Integer division + truncation |
| 4 | `REMAINDER` | Native SFPU | `a - floor(a/b)*b` | Division + multiply + subtract |
| 5 | `FMOD` | Native SFPU | `a - trunc(a/b)*b` | Division + multiply + subtract |
| 6 | `QUANT` | Native SFPU | `round(a * b) + zp` | Multiply + round + add; runtime zero_point param |
| 7 | `REQUANT` | Native SFPU | `round(a * b) + zp` | Re-quantize: multiply + round + add |
| 8 | `DEQUANT` | Native SFPU | `(a - zp) * b` | De-quantize: subtract + multiply |
| 9 | `WHERE` | Native SFPU | `cond ? a : b` | 3-input conditional select; dedicated kernel |
| 10 | `LOGICAL_AND` | Composite | `nez(nez(a) * nez(b))` | 3 cheap unary steps around MUL |
| 11 | `LOGICAL_OR` | Composite | `nez(nez(a) + nez(b))` | 3 cheap unary steps around ADD |
| 12 | `LOGICAL_XOR` | Composite | `nez(nez(a) - nez(b))` | 3 cheap unary steps around SUB |
| 13 | `DIV` (FPU) | Composite | `a * recip(b)` | RECIP pre (multi-step) + MUL |
| 14 | `BIAS_GELU` | Composite | `gelu(a + b)` | GELU post (multi-step approximation) |

### HARD (4 native + 4 composite = 8 total)

Transcendental functions (log, exp, sqrt), iterative algorithms with data-dependent iteration counts, or chains of multiple expensive operations.

| # | Operation | Type | Math | Why HARD |
|---|---|---|---|---|
| 1 | `POWER` | Native SFPU | `a^b` | `exp(b * log(a))` — 3 transcendental ops |
| 2 | `XLOGY` | Native SFPU | `a * log(b)` (0 if a==0) | log + multiply + conditional zero-check |
| 3 | `GCD` | Native SFPU | `gcd(a, b)` | Iterative Euclidean; variable-latency per element |
| 4 | `LCM` | Native SFPU | `lcm(a, b)` | GCD + multiply + divide; variable-latency |
| 5 | `LDEXP` | Composite | `a * 2^b` | EXP2 transcendental pre-process |
| 6 | `LOGADDEXP` | Composite | `log(exp(a) + exp(b))` | 3 transcendental ops (EXP + EXP + LOG) |
| 7 | `LOGADDEXP2` | Composite | `log2(2^a + 2^b)` | 3 transcendental ops (EXP2 + EXP2 + LOG2) |
| 8 | `HYPOT` | Composite | `sqrt(a^2 + b^2)` | 2 SQUARE pre + SQRT post |

---

## Data Type Support Matrix

| Data Type | Native SFPU Binary | FPU Binary | Notes |
|---|---|---|---|
| **BFloat16** | MAXIMUM, MINIMUM, POWER, XLOGY, FMOD | ADD, SUB, MUL | Most ops decompose via FPU + unary |
| **Float32** | All 28 SfpuBinaryOp values | — | Full native SFPU support |
| **Int32** | ADD, SUB, MUL, RSUB, DIV, DIV_FLOOR, DIV_TRUNC, REMAINDER, FMOD, comparisons, BITWISE_*, SHIFT_*, GCD, LCM, QUANT/REQUANT/DEQUANT, MAX, MIN | — | Integer operations are SFPU-exclusive |
| **UInt32** | ADD, SUB, MUL, RSUB, BITWISE_*, SHIFT_*, GCD, LCM, MAX, MIN | — | Subset of Int32 support |
| **UInt16** | ADD, SUB, MUL, RSUB, BITWISE_* | — | Smallest integer type supported |
