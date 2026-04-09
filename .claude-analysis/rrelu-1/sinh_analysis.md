# SFPU Kernel Analysis: sinh

## 1. Overview

**Operation**: `sinh` (hyperbolic sine)
**Math Definition**: `sinh(x) = (exp(x) - exp(-x)) / 2`
**UnaryOpType Enum**: `UnaryOpType::SINH` (defined in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:35`)
**Parameterized**: No (no runtime parameters)
**Approximation Mode**: `false` (returned by `get_op_approx_mode()` default case in `unary_op_utils.cpp:75`)

## 2. Abstraction Layer Inventory

The sinh operation spans 11 abstraction layers from Python API down to SFPU kernel:

### Layer 1: Python API & Golden Function
- **File**: `ttnn/ttnn/operations/unary.py:41-47`
- **Registration**: `ttnn.attach_golden_function(ttnn.sinh, golden_function=_golden_function_sinh)`
- **Golden function**: `torch.sinh(input_tensor_a)`

### Layer 2: C++ TTNN API Registration
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:116`
- **Macro**: `REGISTER_UNARY_OPERATION(sinh, SINH)`
- This macro expands to an inline function `sinh(...)` that calls `ttnn::detail::unary_impl(...)` with `UnaryWithParam{UnaryOpType::SINH}`.

### Layer 3: UnaryOpType Enum
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:35`
- **Value**: `SINH` (among the standard trigonometric/hyperbolic ops, positioned between `COSH` and `ABS`)

### Layer 4: Op Utils — Macro Definition (Split Include Guard)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:23`
- **Mapping**: `UnaryOpType::SINH` → `"SFPU_OP_SINH_INCLUDE"`
- This define is set to `"1"` at compile time and gates the `#include` in `sfpu_split_includes.h`.

### Layer 5: Op Utils — Init and Func Strings (SFPU_OP_CHAIN)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:52`
- **Init string**: `"sinh_tile_init();"`
- **Func string**: `"sinh_tile({idst});"`
- These strings are injected as `SFPU_OP_CHAIN_0_INIT_0` and `SFPU_OP_CHAIN_0_FUNC_0` preprocessor defines.

### Layer 6: Compute Kernel — eltwise_sfpu.cpp (SFPU_OP_CHAIN_0 dispatch)
- **File (legacy)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **File (unary_ng)**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/compute/eltwise_sfpu.cpp`
- **Mechanism**: The `SFPU_OP_CHAIN_0` macro expands to `sinh_tile_init(); sinh_tile(0);` which runs between `copy_tile` and `tile_regs_commit`.
- **Circular Buffers**: Input from `c_0`, output to `c_2`.

### Layer 7: Split Include Header
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h:20-22`
- **Guard**: `#if SFPU_OP_SINH_INCLUDE` → `#include "api/compute/eltwise_unary/sinh.h"`

### Layer 8: Compute Kernel API Header
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
- **Functions**:
  - `sinh_tile(uint32_t idst)` → calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` under `MATH()` guard (runs on math RISC-V only)
  - `sinh_tile_init()` → calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()` under `MATH()` guard
- **Include guard**: Only compiles on `TRISC_MATH` (math thread).

### Layer 9: LLK Math Wrapper
- **File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- **File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- **Init function**: `llk_math_eltwise_unary_sfpu_sinh_init<APPROXIMATE>()` → calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`
- **Compute function**: `llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE, ITERATIONS=8>(dst_index, vector_mode=RC)` → calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`
- Identical across Wormhole B0 and Blackhole.

### Layer 10: SfpuType Enum
- **File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h:12`
- **File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h:12`
- **Value**: `SfpuType::sinh`

### Layer 11: SFPU Kernel (ckernel_sfpu_sinh.h)
- **File (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- **File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- **Identical across architectures**: Yes (both files are byte-for-byte identical)
- **Namespace**: `ckernel::sfpu`
- **Headers**: `ckernel.h`, `ckernel_defs.h`, `sfpi.h`

## 3. SFPU Kernel Deep Dive

### 3.1 Function Signatures

```cpp
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z);

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh();

template <bool APPROXIMATION_MODE>
inline void sinh_init();
```

### 3.2 Algorithm

The kernel computes `sinh(x) = (exp(x) - exp(-x)) / 2` using a two-regime approach:

**Regime 1: Large |x| (|x| >= 0.5)** — Exponential subtraction:
1. Convert to base-2: `z_pos = x * log2(e)`
2. Clamp `z_pos` to `>= -127.0` to prevent IEEE underflow
3. Compute `exp(x) = 2^z_pos` using `exp_21f()` helper
4. Compute `exp(-x) = 2^(-z_pos)` with same clamping
5. Result: `y = (exp_pos - exp_neg) * 0.5`

**Regime 2: Small |x| (|x| < 0.5)** — Taylor approximation:
1. Compute `y = x + x^3 / 6` (first two terms of Taylor series)
2. This avoids catastrophic cancellation when `exp(x) ≈ exp(-x) ≈ 1.0`

**Final step**: Convert to bfloat16 via `float_to_fp16b(y, 0)` for deterministic rounding.

### 3.3 exp_21f Helper — Fast 2^z Computation

The `exp_21f` function implements the Moroz et al. (2022) algorithm for computing `2^z`:

1. **Scale**: `z = addexp(z, 23)` — multiplies z by 2^23 to shift fractional bits into integer position
2. **Bias**: Add IEEE 754 bias `0x3F800000` (1.0f) and convert to int via `_float_to_int32_positive_`
3. **Decompose**: Extract exponent (`exexp`) and 9-bit mantissa (`exman9`) from the integer representation
4. **Polynomial refinement**: Uses 3 constants to refine `2^frac(z)`:
   - `d1 = 0.40196114e-7f`
   - `d2 = int32_to_float(0xf94ee7 + man_part, 0)`
   - `d3 = int32_to_float(0x560e + man_part, 0)`
   - `frac_int = float_to_int32_positive(d1 * d2 * d3)`
5. **Reconstruct**: `setexp(frac_int, 127 + exp_part)` — combine mantissa fraction with exponent

### 3.4 SFPI Instructions Used

| SFPI Instruction | Usage | Description |
|---|---|---|
| `sfpi::vFloat` | Throughout | SIMD vector float type (32 lanes) |
| `sfpi::vInt` | In `exp_21f` | SIMD vector integer type |
| `sfpi::dst_reg[0]` | Read input / Write output | Access DST register tile elements |
| `sfpi::dst_reg++` | End of iteration | Advance to next face/row |
| `sfpi::addexp(z, 23)` | `exp_21f` step 1 | Add 23 to the exponent field (multiply by 2^23) |
| `_float_to_int32_positive_` | `exp_21f` steps 2,4 | Convert float to int (positive values only) |
| `sfpi::exexp()` | `exp_21f` step 3 | Extract exponent from float bit pattern |
| `sfpi::exman9()` | `exp_21f` step 3 | Extract 9-bit mantissa from float bit pattern |
| `sfpi::int32_to_float()` | `exp_21f` step 4 | Convert integer to float with exponent offset |
| `sfpi::setexp()` | `exp_21f` step 5 | Set exponent field of a float |
| `sfpi::reinterpret<>()` | Multiple | Bitwise reinterpret between vFloat and vInt |
| `sfpi::setsgn(x, 0)` | Small-|x| check | Clear sign bit (absolute value) |
| `sfpi::float_to_fp16b(y, 0)` | Final rounding | Convert float32 to bfloat16 for deterministic output |
| `v_if` / `v_endif` | Clamping, regime select | SIMD predicated execution (per-lane branching) |
| Arithmetic: `*`, `+`, `-` | Throughout | SFPU vector multiply, add, subtract |
| Negation: `-z_pos` | Compute z_neg | SFPU vector negate |

### 3.5 Constants

| Constant | Value | Purpose |
|---|---|---|
| `log2e` | `1.4426950408889634f` | Conversion factor: `log2(e)` for `exp(x) = 2^(x * log2e)` |
| `v_half` | `0.5f` | Division by 2 in `(exp(x)-exp(-x))/2`; also regime threshold |
| `v_low_threshold` | `-127.0f` | Clamping floor for `z` to prevent IEEE underflow in `2^z` |
| `v_sixth` | `0.16666667f` | `1/6` for Taylor term `x^3/6` |
| `bias` (in exp_21f) | `0x3F800000` (1.0f) | IEEE 754 bias for 2^z reconstruction |
| `d1` (in exp_21f) | `0.40196114e-7f` | Polynomial coefficient for mantissa refinement |
| `0xf94ee7` (in exp_21f) | integer | Polynomial coefficient offset for d2 |
| `0x560e` (in exp_21f) | integer | Polynomial coefficient offset for d3 |

### 3.6 Iteration Structure

- **ITERATIONS**: Default 8 (processes 8 faces per tile)
- **Loop**: `#pragma GCC unroll 0` prevents unrolling (keeps code size small on RISC-V)
- **Per-iteration**: Read `dst_reg[0]`, compute sinh, write `dst_reg[0]`, advance `dst_reg++`

### 3.7 Initialization

```cpp
template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

The init function is empty — no LREG configuration or programmable constant setup required. This is typical for operations that use inline constants rather than SFPU programmable registers.

## 4. Numerical Considerations

### 4.1 Small-x Regime (|x| < 0.5)
The Taylor approximation `sinh(x) ≈ x + x^3/6` is used when `|x| < 0.5` to avoid catastrophic cancellation. The kernel comments state this is accurate to `< 1 ULP` in bfloat16 for this range.

### 4.2 Large-x Clamping
The exponent argument `z` is clamped to `>= -127.0f` before passing to `exp_21f()`. This prevents IEEE 754 underflow (minimum normal exponent for single precision is -126; -127 produces subnormals).

### 4.3 Bfloat16 Rounding
The final result is explicitly converted to bfloat16 via `float_to_fp16b(y, 0)` then reinterpreted back to vFloat. This ensures deterministic rounding behavior regardless of intermediate precision.

### 4.4 Overflow Behavior
For very large `|x|`, `exp(x)` overflows to infinity while `exp(-x)` underflows to zero (or vice versa for negative x). The result is `±inf * 0.5 = ±inf`, which is the correct IEEE behavior for `sinh(±large)`.

## 5. Cross-Architecture Portability

The Wormhole B0 and Blackhole implementations are **identical** — both `ckernel_sfpu_sinh.h` and `llk_math_eltwise_unary_sfpu_sinh.h` files are byte-for-byte the same across architectures. This operation relies only on standard SFPI intrinsics available on both platforms.

## 6. Registration in unary_ng (Next-Gen Unary Pipeline)

The sinh operation is also registered in the `unary_ng` pipeline:
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp:23,90`
- **Split include**: Same `SFPU_OP_SINH_INCLUDE` guard
- **Init/func**: Same `sinh_tile_init()` / `sinh_tile({idst})` strings
- **Compute kernel**: Same `eltwise_sfpu.cpp` mechanism with `SFPU_OP_CHAIN_0` dispatch

## 7. File Manifest

### New files (operation-specific)
| File | Layer | Purpose |
|---|---|---|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` | 11 | SFPU kernel: `calculate_sinh()`, `exp_21f()`, `sinh_init()` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` | 11 | SFPU kernel (identical to WH B0) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` | 9 | LLK math wrapper |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` | 9 | LLK math wrapper (identical to WH B0) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` | 8 | Compute kernel API: `sinh_tile()`, `sinh_tile_init()` |

### Modified files (shared infrastructure, contain sinh entries)
| File | Layer | What was added |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | 3 | `SINH` in `UnaryOpType` enum |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | 4,5 | `SFPU_OP_SINH_INCLUDE` define, init/func strings, `string_to_unary_with_param` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | 2 | `REGISTER_UNARY_OPERATION(sinh, SINH)` |
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` | 4,5 | `SFPU_OP_SINH_INCLUDE` define, init/func strings (unary_ng) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | 7 | `#if SFPU_OP_SINH_INCLUDE` include guard |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` | 10 | `sinh` in `SfpuType` enum |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | 10 | `sinh` in `SfpuType` enum |
| `ttnn/ttnn/operations/unary.py` | 1 | Golden function + attachment |

## 8. Key Patterns for Reuse

### Pattern: Non-parameterized unary op with custom SFPU kernel
- **No parameters** → Uses `get_op_init_and_func_default()`, no entry in `get_op_init_and_func_parameterized()`
- **Custom include guard** → Needs entry in `get_macro_definition()` returning `"SFPU_OP_<NAME>_INCLUDE"`
- **Split include** → Add `#if SFPU_OP_<NAME>_INCLUDE` / `#include` block in `sfpu_split_includes.h`
- **Dedicated kernel file** → `ckernel_sfpu_<name>.h` in both `wormhole_b0` and `blackhole` directories
- **Empty init** → When no programmable constants or LREG setup is needed

### Pattern: Two-regime computation for numerical stability
- Use Taylor expansion near zero to avoid catastrophic cancellation
- Use standard formula for large values
- Switch via `v_if(abs_x < threshold)` predicated execution

### Pattern: exp_21f base-2 exponential
- Reusable helper for any operation needing `exp(x)` via `2^(x * log2e)`
- Uses Moroz et al. fast polynomial approximation
- Requires clamping input to `[-127, ...]` to prevent underflow

### Pattern: bfloat16 deterministic rounding
- `y = reinterpret<vFloat>(float_to_fp16b(y, 0))` at the end of computation
- Ensures consistent output precision
