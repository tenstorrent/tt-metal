# SFPU Kernel Analysis: sinh

## 1. Operation Identity

| Property | Value |
|----------|-------|
| **Operation name** | sinh |
| **UnaryOpType enum** | `UnaryOpType::SINH` |
| **SfpuType enum** | `SfpuType::sinh` |
| **Math definition** | `sinh(x) = (exp(x) - exp(-x)) / 2` |
| **Parameters** | None (not parametrized) |
| **Approximation mode** | Template parameter `APPROXIMATION_MODE`, not used in current implementation (both paths identical) |
| **Supported dtypes** | BFLOAT16, BFLOAT8_B, FLOAT32 |
| **Supported range** | [-9, 9] (documented in nanobind) |

## 2. File Inventory

### Layer 1: SFPU Kernel (ckernel)
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- Content is **identical** across both architectures.
- Defines `calculate_sinh<APPROXIMATION_MODE, ITERATIONS>()` and `sinh_init<APPROXIMATION_MODE>()` in `namespace ckernel::sfpu`.
- Also defines a private helper `exp_21f<APPROXIMATION_MODE>(vFloat z)` for computing `2^z`.

### Layer 2: LLK Math Wrapper
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- Content is **identical** across both architectures.
- Defines `llk_math_eltwise_unary_sfpu_sinh_init<APPROXIMATE>()` and `llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE, ITERATIONS>(dst_index, vector_mode)`.
- Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`.
- Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_sinh<...>, dst_index, vector_mode)`.

### Layer 3: SfpuType Enum
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- Both contain `SfpuType::sinh` as an enum member.

### Layer 4: Compute API (tile-level)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
- Defines `sinh_tile(uint32_t idst)` and `sinh_tile_init()`.
- Guarded by `#ifdef TRISC_MATH`, dispatches to `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`.

### Layer 5: Split Include Guard
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- Contains `#if SFPU_OP_SINH_INCLUDE` / `#include "api/compute/eltwise_unary/sinh.h"`.

### Layer 6: UnaryOpType Enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- Contains `SINH` in the `UnaryOpType` enum (line 35).

### Layer 7: Op Utils (dispatch glue)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
  - `get_macro_definition`: `SINH` -> `"SFPU_OP_SINH_INCLUDE"`
  - `get_op_init_and_func_default`: `SINH` -> `{"sinh_tile_init();", "sinh_tile({idst});"}`
  - `string_to_unary_with_param`: `"sinh"` -> `UnaryWithParam(UnaryOpType::SINH)`
  - `is_parametrized_type`: SINH is NOT parametrized (returns false via default).
  - `get_op_approx_mode`: Returns `false` (default case).
  - `get_compute_kernel_path`: Returns `"eltwise_sfpu.cpp"` (default case).

### Layer 8: Unary NG (next-gen) Op Utils
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
  - Same `get_macro_definition` and `get_op_init_and_func_default` mappings as Layer 7.

### Layer 9: C++ API Registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
  - `REGISTER_UNARY_OPERATION(sinh, SINH)` at line 116.
  - Generates `ttnn::sinh(input_tensor, memory_config, optional_output_tensor, sub_core_grids)`.

### Layer 10: Python Nanobind
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
  - `bind_unary_operation<"sinh", &ttnn::sinh>(mod, ...)` at line 1791.
  - Math doc: `\sinh(\mathrm{input\_tensor}_i)`
  - Supported range: `[-9, 9]`
  - Supported dtypes: `BFLOAT16, BFLOAT8_B, FLOAT32`

### Layer 11: Python Golden Function
- `ttnn/ttnn/operations/unary.py`
  - `_golden_function_sinh` uses `torch.sinh`.
  - `ttnn.attach_golden_function(ttnn.sinh, golden_function=_golden_function_sinh)`.

### Backward Operation
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp`
  - `sinh_bw(grad, input, output_mem_config)`: Computes `grad * cosh(input)`, with infinity handling for large inputs.

### Tests
- `tests/ttnn/unit_tests/operations/eltwise/unary/test_sinh.py` — Exhaustive bfloat16 bitpattern test with ULP and allclose assertions.
- `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py` — Basic shape-parametrized functional test.

## 3. SFPU Kernel Deep Dive

### 3.1 Algorithm Overview

The `calculate_sinh` function implements sinh using two regimes:

1. **Large |x| regime** (|x| >= 0.5): Direct computation via `(exp(x) - exp(-x)) / 2`, where exp is computed as `2^(x * log2(e))` using a custom `exp_21f` helper.

2. **Small |x| regime** (|x| < 0.5): Taylor series approximation `sinh(x) ~ x + x^3/6` to avoid catastrophic cancellation when `exp(x) ~ exp(-x) ~ 1.0`.

### 3.2 The `exp_21f` Helper

This is a fast `2^z` implementation based on the Moroz et al. 2022 algorithm. It operates in 5 steps:

1. **Scale**: `z = addexp(z, 23)` — multiplies z by 2^23 to shift the fractional part into integer bits.
2. **Bias + convert**: Adds IEEE 754 bias (0x3F800000 = 1.0f), converts to int with `_float_to_int32_positive_`.
3. **Decompose**: Extracts exponent and mantissa via `exexp()` and `exman9()`.
4. **Polynomial refinement**: Computes `2^frac(z)` using a degree-2 polynomial with magic constants:
   - `d1 = 0.40196114e-7f`
   - `d2 = int32_to_float(0xf94ee7 + mantissa, 0)`
   - `d3 = int32_to_float(0x560e + mantissa, 0)`
   - `result_frac = float_to_int(d1 * d2 * d3)`
5. **Reconstruct**: `setexp(frac, 127 + exp_part)` reassembles the IEEE 754 float.

### 3.3 Main `calculate_sinh` Flow

```
for each tile row (d = 0..ITERATIONS-1):
    x = dst_reg[0]                          // read input from DST

    z_pos = x * log2(e)                     // convert to base-2
    clamp z_pos >= -127                     // prevent underflow
    exp_pos = exp_21f(z_pos)                // compute exp(x)

    z_neg = -z_pos                          // negate for exp(-x)
    clamp z_neg >= -127                     // prevent underflow
    exp_neg = exp_21f(z_neg)                // compute exp(-x)

    y = (exp_pos - exp_neg) * 0.5           // sinh formula

    if |x| < 0.5:                           // small-x override
        y = x + x*x*x * (1/6)              // Taylor: x + x^3/6

    y = float_to_fp16b(y, 0)               // round to bfloat16
    dst_reg[0] = y                          // write result
    dst_reg++                               // advance to next row
```

### 3.4 SFPI Instructions Used

| SFPI Instruction | Usage | Purpose |
|-----------------|-------|---------|
| `sfpi::addexp(z, 23)` | `exp_21f` step 1 | Adds 23 to the exponent field, effectively multiplying by 2^23 |
| `_float_to_int32_positive_(v)` | `exp_21f` steps 2, 4 | Converts float to positive int32 (truncation toward zero) |
| `sfpi::exexp(v)` | `exp_21f` step 3 | Extracts the exponent field of an IEEE 754 float |
| `sfpi::exman9(v)` | `exp_21f` step 3 | Extracts the 9-bit mantissa portion |
| `sfpi::int32_to_float(v, 0)` | `exp_21f` step 4 | Converts int32 to float with specified exponent bias |
| `sfpi::setexp(v, exp)` | `exp_21f` step 5 | Sets the exponent field of a float to a given value |
| `sfpi::reinterpret<T>(v)` | Multiple | Bitwise reinterpretation between vFloat and vInt |
| `sfpi::setsgn(x, 0)` | Main loop | Clears the sign bit to compute absolute value |
| `sfpi::float_to_fp16b(y, 0)` | Main loop | Rounds result to bfloat16 for deterministic output |
| `sfpi::dst_reg[0]` | Main loop | Reads/writes the DST register (SFPU data register) |
| `v_if / v_endif` | Main loop | SFPI conditional execution (per-lane predication) |

### 3.5 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `log2e` | `1.4426950408889634f` | log2(e), converts natural log base to base-2 |
| `v_half` | `0.5f` | Divisor for `(exp(x)-exp(-x))/2` and threshold for small-x regime |
| `v_low_threshold` | `-127.0f` | Clamping floor to prevent 2^z underflow (IEEE 754 min exponent) |
| `v_sixth` | `0.16666667f` | 1/6, coefficient for x^3 term in Taylor expansion |
| `0x3f800000` | IEEE 754 `1.0f` | Bias constant in exp_21f |
| `0.40196114e-7f` | Polynomial coeff | Moroz algorithm constant for 2^frac refinement |
| `0xf94ee7` | Polynomial magic | Integer offset for mantissa-based polynomial evaluation |
| `0x560e` | Polynomial magic | Integer offset for mantissa-based polynomial evaluation |

### 3.6 Init Function

```cpp
template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

The init is empty — no LREG (local register) programming is required. This differs from some operations (like exp or sigmoid) that pre-load constants into LREGs.

### 3.7 Iteration Count

Default `ITERATIONS = 8`, which processes 8 rows of the 32x32 tile (each row is 32 elements processed in SIMD). 8 iterations * 4 SFPU lanes * 32 elements/row = 1024 elements = one full tile.

### 3.8 Numerical Stability Strategy

1. **Underflow clamping**: Both `z_pos` and `z_neg` are clamped to >= -127.0, preventing 2^z from producing denormals or zero when the mathematically correct exp result is negligibly small.

2. **Catastrophic cancellation avoidance**: For |x| < 0.5, `exp(x)` and `exp(-x)` are both close to 1.0, making their difference lose significant bits. The Taylor approximation `x + x^3/6` is used instead, which is accurate to < 1 ULP in bfloat16 for this range.

3. **bfloat16 rounding**: The final `float_to_fp16b` ensures deterministic rounding behavior, preventing platform-dependent intermediate precision from affecting results.

## 4. Dispatch Chain Summary

```
Python: ttnn.sinh(tensor)
  -> C++ ttnn::sinh(tensor, ...) [via REGISTER_UNARY_OPERATION macro]
    -> ttnn::detail::unary_impl(tensor, {UnaryWithParam{UnaryOpType::SINH}}, ...)
      -> UnaryProgramFactory creates compute kernel with defines:
           SFPU_OP_CHAIN_0_INIT_0 = "sinh_tile_init();"
           SFPU_OP_CHAIN_0_FUNC_0 = "sinh_tile(0);"
           SFPU_OP_SINH_INCLUDE = "1"
        -> Kernel: eltwise_sfpu.cpp
          -> sinh_tile(idst)
            -> MATH(llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst))
              -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_sinh<...>, idst, vector_mode)
                -> calculate_sinh<APPROX, 8>()  [SFPU execution]
```

## 5. Key Design Patterns

1. **Split include guard pattern**: `SFPU_OP_SINH_INCLUDE` macro controls whether `sinh.h` is compiled into the compute kernel. This reduces kernel binary size by only including needed operations.

2. **Non-parametrized unary op**: sinh takes no runtime parameters. It uses `get_op_init_and_func_default` (not the parametrized path). No runtime args are passed to the kernel.

3. **Private exp helper**: The `exp_21f` function is defined locally inside `ckernel_sfpu_sinh.h` rather than shared. This avoids cross-include dependencies at the cost of potential duplication if another operation uses the same algorithm.

4. **Dual-regime computation**: The small-x Taylor fallback is a common pattern for hyperbolic functions to maintain accuracy where the primary formula would lose precision.

5. **Architecture-identical kernels**: Wormhole B0 and Blackhole have byte-identical SFPU kernels, indicating the SFPI ISA is the same across these generations for this operation.

## 6. Test Strategy

### Exhaustive bfloat16 test (`tests/ttnn/unit_tests/operations/eltwise/unary/test_sinh.py`)
- Generates all 65536 bfloat16 bit patterns.
- Tests both bfloat16 and float32 input dtypes.
- Filters NaN/Inf before comparison.
- **bfloat16 mode**: ULP threshold = 2, plus allclose with rtol=1.6e-2, atol=1e-2.
- **float32 mode**: allclose with rtol=1.6e-2, atol=1e-2 (no ULP check, since SFPU computes at bf16 precision).

### Basic functional test (`tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`)
- Tests shapes [1,1,32,32], [1,1,64,64], [1,1,32,256].
- Input range: linspace(-4.0, 4.0).
- Tolerance: atol=0.2, rtol=0.05.
