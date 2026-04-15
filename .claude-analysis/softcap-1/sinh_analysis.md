# SFPU Kernel Analysis: `sinh`

## 1. Operation Identity

| Field | Value |
|---|---|
| **Operation name** | `sinh` |
| **UnaryOpType enum** | `UnaryOpType::SINH` |
| **SfpuType enum** | `SfpuType::sinh` |
| **Math definition** | `sinh(x) = (exp(x) - exp(-x)) / 2` |
| **Parameterized** | No |
| **Approx mode** | Always `false` (not in `get_op_approx_mode` switch) |
| **Python API** | `ttnn.sinh(input_tensor)` |
| **Golden function** | `torch.sinh(input_tensor)` |
| **Supported dtypes** | BFLOAT16, BFLOAT8_B, FLOAT32 |
| **Supported range** | [-9, 9] (documented in nanobind) |

## 2. File Inventory

### SFPU Kernel (lowest level)
| File | Purpose |
|---|---|
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` | Blackhole SFPU kernel: `calculate_sinh()`, `sinh_init()`, `exp_21f()` helper |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` | Wormhole B0 SFPU kernel (identical to Blackhole) |

### LLK Math Wrapper
| File | Purpose |
|---|---|
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` | Blackhole LLK wrapper: `llk_math_eltwise_unary_sfpu_sinh()`, `llk_math_eltwise_unary_sfpu_sinh_init()` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` | Wormhole B0 LLK wrapper (identical to Blackhole) |

### SfpuType Registration
| File | Purpose |
|---|---|
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | Declares `SfpuType::sinh` enum value |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` | Same for Wormhole B0 |

### Compute API (tile-level)
| File | Purpose |
|---|---|
| `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` | Exposes `sinh_tile(idst)` and `sinh_tile_init()` to compute kernels |
| `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Conditional include: `#if SFPU_OP_SINH_INCLUDE` → `sinh.h` |

### Host-side Registration
| File | Purpose |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::SINH` enum value |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Maps `SINH` to `sinh_tile_init()/sinh_tile()` defines, sets `SFPU_OP_SINH_INCLUDE=1` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | `REGISTER_UNARY_OPERATION(sinh, SINH)` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | Python binding via `bind_unary_operation<"sinh", &ttnn::sinh>(...)` |
| `ttnn/ttnn/operations/unary.py` | Golden function attachment: `ttnn.attach_golden_function(ttnn.sinh, ...)` |

### Unary NG (next-gen) Pathway
| File | Purpose |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` | Maps `SINH` → `"SFPU_OP_SINH_INCLUDE"` and `sinh_tile_init()/sinh_tile()` |

### Tests
| File | Purpose |
|---|---|
| `tests/ttnn/unit_tests/operations/eltwise/unary/test_sinh.py` | Exhaustive bfloat16 bitpattern test (ULP + allclose) |
| `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py` | Basic functional test (linspace -4..4) |
| `tests/sweep_framework/sweeps/eltwise/unary/sinh/sinh.py` | Sweep test (various shapes, dtypes, memory configs) |
| `tests/sweep_framework/sweeps/eltwise/unary/sinh/sinh_sharded.py` | Sharded sweep test |

## 3. SFPU Kernel Deep Dive

### 3.1. Kernel Function Signature

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh();
```

- **APPROXIMATION_MODE**: Template parameter threaded through from `APPROX` macro, always `false` for `sinh` (per `get_op_approx_mode`). Currently unused in the kernel body — the algorithm is the same regardless.
- **ITERATIONS**: Default 8 (processes 8 tile faces per call). Standard for unary SFPU ops.

### 3.2. Init Function

```cpp
template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

Empty — no LREG configuration required. This is typical for operations that don't use lookup tables or SFPU-programmable constants.

### 3.3. Algorithm

The kernel implements a two-regime piecewise computation:

#### Regime 1: Large |x| (|x| >= 0.5)
Uses the identity `sinh(x) = (exp(x) - exp(-x)) / 2`, computed via base-2 exponentiation:

1. Convert to base-2: `z_pos = x * log2(e)` where `log2(e) = 1.4426950408889634`
2. Clamp `z_pos` to `>= -127.0` to prevent underflow in `exp_21f`
3. Compute `exp_pos = 2^z_pos` via `exp_21f()` helper
4. Negate: `z_neg = -z_pos`, clamp similarly
5. Compute `exp_neg = 2^z_neg` via `exp_21f()` helper
6. Result: `y = (exp_pos - exp_neg) * 0.5`

#### Regime 2: Small |x| (|x| < 0.5)
Taylor approximation to avoid catastrophic cancellation:

```
sinh(x) ≈ x + x³/6
```

When |x| < 0.5, `exp(x)` and `exp(-x)` are both close to 1.0, so their difference loses precision. The Taylor series third-order approximation is accurate to < 1 ULP in bfloat16 for this range.

#### Final Step
The result is explicitly rounded to bfloat16 via `float_to_fp16b(y, 0)` for deterministic rounding behavior.

### 3.4. Helper Function: `exp_21f`

A custom implementation of `2^z` using the Moroz et al. 2022 ("exp_21f") algorithm. This is **defined locally** within `ckernel_sfpu_sinh.h` (not shared from a common exp header).

```cpp
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z);
```

**Algorithm steps:**
1. Scale input: `z = addexp(z, 23)` — multiply by 2^23 to shift fractional bits into integer position
2. Add IEEE 754 bias (0x3F800000 = 1.0f representation) and convert to int
3. Decompose into exponent and mantissa: `exexp()` + `exman9()`
4. Polynomial refinement for 2^frac(z) using hardcoded coefficients:
   - `d1 = 0.40196114e-7f`
   - Mantissa adjustments: `0xf94ee7`, `0x560e`
5. Reconstruct: `setexp(frac_result, 127 + exp_part)`

### 3.5. SFPI Instructions Used

| Instruction | Usage | Purpose |
|---|---|---|
| `sfpi::addexp(z, 23)` | `exp_21f` step 1 | Add 23 to the exponent of `z` (multiply by 2^23) |
| `_float_to_int32_positive_()` | `exp_21f` steps 2, 4 | Convert positive float to int32 (truncation) |
| `sfpi::exexp()` | `exp_21f` step 3 | Extract exponent field from float |
| `sfpi::exman9()` | `exp_21f` step 3 | Extract 9-bit mantissa from float |
| `sfpi::int32_to_float()` | `exp_21f` step 4 | Convert int32 to float with exponent offset |
| `sfpi::setexp()` | `exp_21f` step 5 | Set exponent field of a float |
| `sfpi::reinterpret<>()` | Throughout | Bitwise reinterpret between vFloat/vInt |
| `sfpi::setsgn(x, 0)` | Main loop | Clear sign bit (absolute value) |
| `sfpi::float_to_fp16b(y, 0)` | Main loop | Convert to bfloat16 (deterministic rounding) |
| `sfpi::dst_reg[0]` | Main loop | Read from / write to DST register |
| `v_if / v_endif` | Main loop | SFPU conditional execution (predicated lanes) |

### 3.6. Constants

| Constant | Value | Purpose |
|---|---|---|
| `log2e` | `1.4426950408889634f` | `log2(e)` for base conversion: `exp(x) = 2^(x * log2(e))` |
| `v_half` | `0.5f` | Division by 2 in sinh formula; also threshold for Taylor regime |
| `v_low_threshold` | `-127.0f` | Minimum exponent clamp to prevent underflow in 2^z |
| `v_sixth` | `0.16666667f` | `1/6` coefficient for Taylor term `x³/6` |
| `0x3f800000` | IEEE 754 `1.0f` | Bias for integer conversion in `exp_21f` |
| `0.40196114e-7f` | Polynomial coeff | `exp_21f` refinement coefficient `d1` |
| `0xf94ee7` | Integer constant | `exp_21f` mantissa adjustment for `d2` |
| `0x560e` | Integer constant | `exp_21f` mantissa adjustment for `d3` |

### 3.7. Tile Iteration Pattern

```cpp
#pragma GCC unroll 0  // Disable unrolling to reduce code size
for (int d = 0; d < ITERATIONS; d++) {
    sfpi::vFloat x = sfpi::dst_reg[0];  // Read current face
    // ... compute ...
    sfpi::dst_reg[0] = y;               // Write result
    sfpi::dst_reg++;                     // Advance to next face
}
```

Standard single-buffered pattern: read face → compute → write face → advance. No circular buffer interaction at this level (CB management is handled by the UnaryProgramFactory).

## 4. Dispatch Chain

The complete dispatch from Python to SFPU kernel:

```
ttnn.sinh(tensor)
  → C++ ttnn::sinh (registered via REGISTER_UNARY_OPERATION)
    → UnaryProgramFactory (standard unary program factory)
      → Compute kernel define generation:
          SFPU_OP_SINH_INCLUDE = 1
          SFPU_OP_CHAIN_0_INIT_0 = "sinh_tile_init();"
          SFPU_OP_CHAIN_0_FUNC_0 = "sinh_tile(i);"
        → Compile-time: #if SFPU_OP_SINH_INCLUDE includes sinh.h
          → sinh_tile(idst) macro wraps:
            MATH(llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst))
              → _llk_math_eltwise_unary_sfpu_params_<false>(
                    ckernel::sfpu::calculate_sinh<false, 8>,
                    dst_index, vector_mode)
                  → calculate_sinh<false, 8>()  [SFPU kernel]
```

## 5. Architecture Compatibility

The Blackhole and Wormhole B0 implementations are **identical** at all levels:
- `ckernel_sfpu_sinh.h` — same algorithm, same constants, same SFPI instructions
- `llk_math_eltwise_unary_sfpu_sinh.h` — same wrapper structure

This indicates the operation does not use any architecture-specific SFPU features.

## 6. Numerical Properties

| Property | Value |
|---|---|
| **Small-x threshold** | `|x| < 0.5` (switches to Taylor) |
| **Taylor terms** | `x + x³/6` (3rd order) |
| **Overflow clamp** | `z >= -127` (prevents 2^z underflow) |
| **Output rounding** | Explicit `float_to_fp16b` for deterministic bf16 |
| **Test ULP tolerance** | 2 ULP (bfloat16 exhaustive test) |
| **Test allclose tolerance** | `rtol=1.6e-2, atol=1e-2` |
| **Safe input range** | [-9, 9] (per sweep tests) |

The Taylor branch (`|x| < 0.5`) is critical for accuracy. Without it, the subtraction `exp(x) - exp(-x)` when both are near 1.0 would cause catastrophic cancellation, potentially losing all significant bits.

## 7. Key Design Decisions

1. **Self-contained `exp_21f` helper**: Rather than calling a shared exponential SFPU function, the kernel defines its own `exp_21f` locally. This avoids cross-header dependencies and allows the kernel to be included independently via the `SFPU_OP_SINH_INCLUDE` split-include mechanism.

2. **Two-regime approach**: The piecewise computation (Taylor for small x, exp-based for large x) is a standard numerical technique. The threshold of 0.5 is well-chosen for bfloat16 precision.

3. **No init required**: `sinh_init()` is empty — no SFPU programmable constants or lookup tables are needed since all constants are embedded as immediates in the kernel code.

4. **No parameterization**: `sinh` takes no runtime parameters. It is not in the `is_parametrized_type` list, so the host-side dispatch always calls `get_op_init_and_func_default`.

5. **Explicit bf16 rounding**: The `float_to_fp16b` call at the end ensures deterministic rounding regardless of the SFPU's internal precision mode.
