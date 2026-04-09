# SFPU Kernel Analysis: sinh

## 1. Operation Overview

**Math definition**: `sinh(x) = (exp(x) - exp(-x)) / 2`

**Operation type**: Unary, non-parametrized (no runtime parameters)

**UnaryOpType enum value**: `SINH` (defined in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:35`)

**Supported data types**: BFLOAT16, BFLOAT8_B, FLOAT32

**Approximation mode**: Always `false` (returns false from `get_op_approx_mode`)

## 2. File Inventory

### Layer 1: SFPU Kernel Function (ckernel)
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- Files are **identical** across both architectures.

### Layer 2: LLK Math Wrapper
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
- Files are **identical** across both architectures.

### Layer 3: Compute API (tile-level)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`

### Layer 4: Split-include guard
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (macro `SFPU_OP_SINH_INCLUDE`)

### Layer 5: SfpuType enum (LLK test infra)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (enum value `sinh`)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (enum value `sinh`)

### Layer 6: Host-side dispatch (op_utils)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

### Layer 7: C++ API registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:116` (`REGISTER_UNARY_OPERATION(sinh, SINH)`)

### Layer 8: Python bindings
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1791-1795`
- `ttnn/ttnn/operations/unary.py:41-47` (golden function: `torch.sinh`)

### Layer 9: Backward operation
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.hpp:275` (`sinh_bw`)
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp:1058-1070`
- Backward: `grad * cosh(input)`, with inf handling for large inputs

### Layer 10: Tests
- `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py` (exhaustive bfloat16 bitpattern test)
- `tests/ttnn/unit_tests/operations/eltwise/unary/test_sinh.py` (basic shape tests)
- `tests/sweep_framework/sweeps/eltwise/unary/sinh/sinh.py` (sweep test)
- `tests/sweep_framework/sweeps/eltwise/unary/sinh/sinh_sharded.py` (sharded sweep)
- `tests/sweep_framework/sweeps/eltwise/unary_backward/sinh_bw/sinh_bw.py` (backward sweep)

## 3. SFPU Kernel Deep Dive

### 3.1 Namespace and Template Signature

```cpp
namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh();

template <bool APPROXIMATION_MODE>
inline void sinh_init();
}
```

- `APPROXIMATION_MODE`: Template parameter (always `false` for this op, passed through from `APPROX` macro)
- `ITERATIONS`: Defaults to 8 (standard for processing all 8 sub-tiles in a tile face). Each iteration processes one 32-element SFPU vector lane.

### 3.2 Init Function

```cpp
template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

The init function is empty. sinh does not require any LREG configuration or programmable constant setup (unlike operations like `exp` which may load LUT constants).

### 3.3 Algorithm Strategy: Two-regime approach

The kernel uses a **two-regime approach** to maximize accuracy:

**Regime 1: Large |x| (|x| >= 0.5)** - Direct exponential computation:
```
sinh(x) = (exp(x) - exp(-x)) / 2
        = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
```

**Regime 2: Small |x| (|x| < 0.5)** - Taylor series approximation:
```
sinh(x) ~= x + x^3/6
```

**Rationale**: For small |x|, `exp(x)` and `exp(-x)` are both close to 1.0, so their subtraction suffers catastrophic cancellation. The Taylor approximation `x + x^3/6` is accurate to < 1 ULP in bfloat16 for |x| < 0.5.

### 3.4 Helper Function: exp_21f

The kernel defines a local helper `exp_21f<APPROXIMATION_MODE>(vFloat z)` that computes `2^z` using the Moroz et al. 2022 algorithm. This is a **self-contained helper defined in the same header**, not imported from a shared library.

```cpp
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa
    sfpi::vInt exp_part = sfpi::exexp(reinterpret<vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(reinterpret<vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);
    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result
    sfpi::vInt result_int = reinterpret<vInt>(setexp(reinterpret<vFloat>(frac_int), 127U + exp_part));
    return reinterpret<vFloat>(result_int);
}
```

**Key algorithm properties**:
- Uses integer bit-manipulation for fast 2^z computation
- Decomposes z into integer exponent + fractional mantissa
- Applies a degree-2 polynomial refinement for the fractional part
- Avoids the shared ckernel `calculate_exponential_body` or LREG-based approaches

### 3.5 Main Compute Loop

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_sixth = 0.16666667f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // === Regime 1: Large |x| ===
        sfpi::vFloat z_pos = x * v_log2e;           // x * log2(e)
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }
        v_endif;
        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);

        sfpi::vFloat z_neg = -z_pos;                // -x * log2(e)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }
        v_endif;
        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);

        sfpi::vFloat y = (exp_pos - exp_neg) * v_half;

        // === Regime 2: Small |x| override ===
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0);
        v_if(abs_x < v_half) {
            sfpi::vFloat x_sq = x * x;
            y = x + x_sq * x * v_sixth;     // x + x^3/6
        }
        v_endif;

        // bfloat16 rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
```

### 3.6 SFPI Intrinsics Used

| Intrinsic | Purpose | Count |
|-----------|---------|-------|
| `sfpi::dst_reg[0]` | Read/write destination register | 2 per iter (read + write) |
| `sfpi::dst_reg++` | Advance to next sub-tile | 1 per iter |
| `sfpi::addexp(v, n)` | Add `n` to exponent field of float | 1 per exp_21f call (x2) |
| `_float_to_int32_positive_()` | Convert float to int (positive values) | 2 per exp_21f call (x2) |
| `sfpi::exexp()` | Extract exponent from float as int | 1 per exp_21f call (x2) |
| `sfpi::exman9()` | Extract 9-bit mantissa from float | 1 per exp_21f call (x2) |
| `sfpi::int32_to_float()` | Convert int to float with exponent bias | 2 per exp_21f call (x2) |
| `sfpi::setexp()` | Set exponent field of float | 1 per exp_21f call (x2) |
| `sfpi::reinterpret<T>()` | Reinterpret cast between vFloat/vInt | 4 per exp_21f call (x2) |
| `sfpi::setsgn(v, 0)` | Clear sign bit (absolute value) | 1 per iter |
| `sfpi::float_to_fp16b(v, 0)` | Convert to bfloat16 for deterministic rounding | 1 per iter |
| `v_if / v_endif` | SFPU predicated execution (conditional) | 3 per iter |
| Arithmetic (`*`, `+`, `-`) | Vector multiply/add/sub | ~10 per iter |

### 3.7 Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `log2e` | 1.4426950408889634f | Convert natural log base to log2 base |
| `v_half` | 0.5f | Division by 2 for sinh formula; also the regime threshold |
| `v_low_threshold` | -127.0f | Clamp lower bound for 2^z to prevent underflow |
| `v_sixth` | 0.16666667f | Coefficient 1/6 for Taylor term x^3/6 |
| `0x3f800000` | IEEE 754 for 1.0f | Bias for exp_21f integer conversion |
| `0.40196114e-7f` | Polynomial coefficient d1 | exp_21f refinement |
| `0xf94ee7` | Integer constant for d2 | exp_21f mantissa polynomial |
| `0x560e` | Integer constant for d3 | exp_21f mantissa polynomial |

### 3.8 Numerical Safeguards

1. **Underflow clamping**: Both `z_pos` and `z_neg` are clamped to `-127.0f` minimum before calling `exp_21f`, preventing IEEE underflow in 2^z computation (2^-127 is the smallest normal float).

2. **Catastrophic cancellation avoidance**: For `|x| < 0.5`, the Taylor series `x + x^3/6` replaces the `exp(x) - exp(-x)` subtraction, which would lose precision when both exponentials are near 1.0.

3. **Deterministic rounding**: The result is explicitly converted to bfloat16 via `float_to_fp16b(y, 0)` before writing back. This ensures deterministic rounding behavior regardless of the internal FP32 computation path.

## 4. Dispatch Chain

### 4.1 Host-side flow

1. **Python API**: `ttnn.sinh(tensor)` calls `ttnn::sinh()` (C++ via nanobind)
2. **C++ API**: `REGISTER_UNARY_OPERATION(sinh, SINH)` expands to call `ttnn::detail::unary_impl()` with `UnaryOpType::SINH`
3. **Op chain**: Creates `UnaryWithParam(UnaryOpType::SINH)` (no parameters)
4. **Compute kernel defines**:
   - `SFPU_OP_CHAIN_0_INIT_0` -> `"sinh_tile_init();"`
   - `SFPU_OP_CHAIN_0_FUNC_0` -> `"sinh_tile(0);"`
   - `SFPU_OP_SINH_INCLUDE` -> `"1"` (triggers the split-include guard)
5. **Compute kernel path**: Returns `"eltwise_sfpu.cpp"` (default)

### 4.2 Device-side call chain

```
sinh_tile(idst)                              // Layer 3: compute API
  -> MATH(llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst))   // conditional compile guard
    -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(        // Layer 2: LLK wrapper
         ckernel::sfpu::calculate_sinh<APPROX, 8>,           // function pointer
         dst_index, vector_mode)
      -> calculate_sinh<false, 8>()                          // Layer 1: SFPU kernel
        -> exp_21f<false>(z_pos)                             // helper for 2^z
        -> exp_21f<false>(z_neg)                             // helper for 2^(-z)
```

### 4.3 Init call chain

```
sinh_tile_init()                             // Layer 3
  -> MATH(llk_math_eltwise_unary_sfpu_sinh_init<APPROX>())  // Layer 2
    -> llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROX>()
      -> sinh_init<APPROX>()                                // Layer 1 (no-op)
```

## 5. Architecture Differences

The Wormhole B0 and Blackhole implementations are **byte-for-byte identical** for this operation:
- `ckernel_sfpu_sinh.h`: Same content
- `llk_math_eltwise_unary_sfpu_sinh.h`: Same content

No architecture-specific code paths, conditionals, or SFPI intrinsic differences exist.

## 6. Unary_ng (Next-gen) Path

The `sinh` operation is also registered in the `unary_ng` dispatch path:
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp:23` (macro define)
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp:90` (init/func strings)

Both the legacy and ng paths use the same macro `SFPU_OP_SINH_INCLUDE` and the same compute functions `sinh_tile_init()` / `sinh_tile(idst)`.

## 7. LLK Test Infrastructure Status

- `SfpuType::sinh` exists in `llk_sfpu_types.h` enum
- **NOT wired** into `sfpu_operations.h` `call_sfpu_operation()` switch -- there is no `case SfpuType::sinh:` handler
- This means the LLK-level test infrastructure (`test_zzz_eltwise_unary_sfpu.py`) cannot directly test this op at the LLK level

## 8. Key Implementation Patterns (for re-implementation)

### Pattern: Self-contained exp helper
The kernel defines its own `exp_21f` helper inline rather than depending on a shared exponential function. This avoids LREG/programmable constant dependencies and keeps the kernel self-contained.

### Pattern: Two-regime conditional
Uses `v_if(abs_x < threshold)` to switch between an accurate direct computation and a Taylor fallback for the numerically unstable regime.

### Pattern: Explicit bfloat16 rounding
Calls `float_to_fp16b(y, 0)` before writing to `dst_reg[0]` to ensure output is in bfloat16 format with deterministic rounding.

### Pattern: Symmetric clamping via negation
Instead of computing `x * log2(e)` and `-x * log2(e)` independently, computes `z_pos = x * log2(e)` then `z_neg = -z_pos`. Both are clamped independently against the underflow threshold.

### Pattern: No init dependencies
The init function is empty, meaning no programmable constants, LREGs, or LUTs need to be configured. The kernel is entirely self-contained.

## 9. Estimated Instruction Count per Iteration

Per iteration (one sub-tile of 32 elements):
- 2x `exp_21f` calls: ~2 x 12 SFPU instructions = ~24
- 3x `v_if/v_endif` blocks: ~6 conditional instructions
- ~10 arithmetic operations (multiply, add, subtract)
- ~3 special ops (setsgn, float_to_fp16b, dst_reg access)
- **Total estimate: ~43 SFPU instructions per iteration**
- **Per tile (8 iterations): ~344 SFPU instructions**

## 10. Test Accuracy Requirements

| Test | Metric | Threshold |
|------|--------|-----------|
| `test_sinh.py` (bfloat16 exhaustive) | ULP | 2 |
| `test_sinh.py` (bfloat16 exhaustive) | allclose | rtol=1.6e-2, atol=1e-2 |
| `test_sinh.py` (fp32) | allclose | rtol=1.6e-2, atol=1e-2 |
| `test_sinh.py` (basic shapes) | allclose | atol=0.2, rtol=0.05 |
| sweep `sinh.py` | PCC | 0.999 |

## 11. Summary

The `sinh` SFPU kernel is a well-structured, self-contained unary operation that:
- Defines its own `exp_21f` helper for `2^z` computation (Moroz et al. 2022 algorithm)
- Uses a two-regime approach: direct exp for large |x|, Taylor series for small |x|
- Requires no init configuration (empty init function)
- Is non-parametrized (no runtime arguments)
- Uses the `SFPU_OP_SINH_INCLUDE` split-include macro for conditional compilation
- Has identical implementations across Wormhole B0 and Blackhole architectures
- Is registered in both legacy unary and unary_ng dispatch paths
