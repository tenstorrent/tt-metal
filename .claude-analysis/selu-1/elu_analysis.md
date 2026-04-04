## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `ELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `elu_tile(0)` (expected; compute API and LLK dispatch layers are **not yet wired** — see note below)

> **Note — Incomplete Dispatch Stack**: The ELU SFPU kernel (`_calculate_elu_` / `_init_elu_`) exists in the tt_llk library (`ckernel_sfpu_elu.h`), but the higher abstraction layers required to invoke it from the TTNN unary pipeline are **not yet implemented**:
> - No `elu_tile()` / `elu_tile_init()` functions in the compute API (`compute_kernel_api.h`)
> - No `llk_math_eltwise_unary_sfpu_elu.h` LLK dispatch header
> - No case for `UnaryOpType::ELU` in `get_op_init_and_func_default()` or `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp`
>
> The `UnaryOpType::ELU` enum value exists in `unary_op_types.hpp`, and the SFPU kernel is ready in tt_llk. The wiring is expected to follow the same pattern as similar operations (e.g., the SELU operation currently being implemented in this worktree).

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ELU)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | Expected: none (default `APPROXIMATION_MODE=false`) | No `get_op_init_and_func` case exists yet; when wired, the init would likely be `elu_tile_init()` with no explicit template param, defaulting to `false` |
| Effective SFPU path | Non-approximate: `_calculate_exponential_piecewise_<false, false, false>` → calls `_sfpu_exp_(setsgn(in, 0))` then conditionally applies `_sfpu_reciprocal_<2>()` for negative inputs | `ckernel_sfpu_exp.h` lines 388–396: the `else` branch of `if constexpr (APPROXIMATION_MODE)` |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist — no `elu_tile()` in `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (expected to be added) |
| **LLK Dispatch** | This level of abstraction doesn't exist — no `llk_math_eltwise_unary_sfpu_elu.h` (expected to be added) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_elu.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared by all unary SFPU ops) |

### Call Chain
The expected call chain (once wired) would be:

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` expands to `elu_tile(0)`.
2. **API Header** (`compute_kernel_api.h`): `elu_tile(idst)` wraps `MATH((llk_math_eltwise_unary_sfpu_elu<APPROX>(idst, slope)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_elu.h`): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_elu_<APPROXIMATE>, dst_index, (int)VectorMode::RC, slope)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces, calling `_calculate_elu_<false, 8>(slope)` per face, with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_elu.h`): `_calculate_elu_` performs the piecewise exponential computation per 8 iterations (one face).

Currently, only steps 4 and 5 exist. Steps 1–3 need to be implemented.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (expected, based on analogous operations) — processes all 4 faces of the tile.
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops over 4 faces (for `VectorMode::RC`). Each iteration calls `_calculate_elu_<APPROXIMATE, 8>(slope)` which processes one face (8 SFPU iterations × 32 elements = 256 elements per face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode is `ADDR_MOD_7` with all increments set to 0 (`.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}`), configured in `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()`. DEST addressing is managed explicitly by `dst_reg++` in the SFPU kernel and `SETRWC` calls with stride 8 in the params dispatch layer.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h

#include <cstdint>
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_elu_(std::uint32_t slope) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    const bool SCALE_EN                       = false; // ELU does not scale input before exp
    const bool SKIP_POSITIVE_CHECK            = false; // ELU does not skip the >= 89 saturation check
    const std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 0x3F80 = 1.0 in BF16

    sfpi::vFloat s = Converter::as_float(slope); // Reinterpret uint32 param as float (the alpha parameter)
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // Load 32 elements from current DEST position

        v_if (v < 0.0f) // Condition code: process only negative elements
        {
            // Compute exp(v) using piecewise exponential; SCALE_EN=false means no pre-scaling
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(v, exp_base_scale_factor);
            v                  = s * (v_exp - 1.0f); // ELU formula: alpha * (exp(x) - 1) for x < 0
        }
        v_endif; // For x >= 0, v is unchanged (identity)

        sfpi::dst_reg[0] = v; // Store result back to DEST

        sfpi::dst_reg++; // Advance by 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_elu_() // APPROXIMATION_MODE=false
{
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in FP32
    const bool FAST_APPROX                    = false; // ELU does not use fast approximation
    // Delegates to exponential init; with APPROX=false, FAST=false, this calls _init_sfpu_reciprocal_<false>()
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}

} // namespace ckernel::sfpu
```

#### Key Helper: `_calculate_exponential_piecewise_` (non-approximate path)

When `APPROXIMATION_MODE=false`, `SCALE_EN=false`, `SKIP_POSITIVE_CHECK=false`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{
    // APPROXIMATION_MODE=false path:
    sfpi::vFloat result = 0.0f;
    // No SCALE_EN scaling (skipped)
    // No APPROXIMATION_MODE branch — goes to else:

    result = _sfpu_exp_(sfpi::setsgn(in, 0)); // Compute exp(|in|) via Horner polynomial + repeated squaring

    v_if (in < 0) // For negative inputs, exp(x) = 1/exp(|x|)
    {
        result = _sfpu_reciprocal_<2>(result); // Newton-Raphson reciprocal with 2 iterations (float32 precision)
    }
    v_endif;

    return result;
}
```

#### Key Helper: `_sfpu_exp_` (Horner polynomial exponential)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // Extract exponent; if >= 0, replace with -1 to normalize to [-1, 0) range
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126); // Set exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Horner form polynomial: exp(x) ≈ 1 + x*(0.8373 + x*0.863281) for x in [-1, 0)
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    // Repeated squaring to recover the integer part of the exponent
    v_if (exp >= 0)
    {
        val = val * val; // Square once unconditionally
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0); // Narrow predication: only continue squaring while exp >= 0
            val = val * val;
        }
    }
    v_endif;

    return val;
}
```

#### Key Helper: `_init_exponential_` (non-approximate, non-fast path)

When `APPROXIMATION_MODE=false`, `FAST_APPROX=false`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h
// (lines 923–927)

    // Falls through to the final else branch:
    // Initialisation for use of _sfpu_reciprocal_<2> in _calculate_exponential_<APPROXIMATION_MODE=false>.
    _init_sfpu_reciprocal_<false>();
    // This sets up polynomial coefficients for the reciprocal initial estimate:
    //   vConstFloatPrgm0 = 0.323232... (k0)
    //   vConstFloatPrgm1 = 1.454545... (k1)
    //   vConstFloatPrgm2 = 2.121212... (k2)
    // Used in: y = k2 - k1*x + k0*x^2 (quadratic initial estimate for 1/x over [1,2))
```

### SFPU Instructions Used

The ELU kernel itself is purely SFPI-abstraction-based. The underlying SFPI abstractions and helper functions emit the following SFPU instructions:

| Instruction / Intrinsic | Description | Used By |
|------------------------|-------------|---------|
| `SFPLOAD` / `dst_reg[0]` read | Load 32 elements from current DEST position into LREG | `_calculate_elu_` — loading input tile data |
| `SFPSTORE` / `dst_reg[0]` write | Store 32 elements from LREG back to DEST | `_calculate_elu_` — writing result |
| `SFPMAD` / `vFloat * vFloat + vFloat` | Fused multiply-add (also emitted for `vFloat + vFloat` as `a * 1.0 + b`) | `_calculate_elu_` — `s * (v_exp - 1.0f)`, Horner polynomial in `_sfpu_exp_`, reciprocal NR iterations |
| `SFPSETCC` / `v_if (v < 0.0f)` | Set condition code based on comparison | `_calculate_elu_` — branch on negative input |
| `SFPENCC` / `v_endif` | End conditional block, restore condition code | `_calculate_elu_`, `_sfpu_exp_`, `_sfpu_reciprocal_` |
| `SFPSETSGN` / `setsgn(in, 0)` | Set sign bit of float to 0 (absolute value) | `_calculate_exponential_piecewise_` — computing exp(|x|) |
| `SFPEXEXP` / `exexp(val)` | Extract exponent from float | `_sfpu_exp_` — extracting integer exponent |
| `SFPSETEXP` / `setexp(val, 126)` | Set exponent field of float | `_sfpu_exp_` — normalizing to [-1, 0) range |
| `SFPSETMAN` / `setman(...)` | Set mantissa field of float | `_sfpu_reciprocal_` — constructing scale factor and initial value |
| `SFPNOT` / `~reinterpret<vUInt>(in)` | Bitwise NOT (computes 255 - exponent efficiently) | `_sfpu_reciprocal_` — computing reciprocal scale factor |
| `SETRWC` | Set read/write counters for DEST address progression | Parameters dispatch — advancing between faces |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST (via `dst_reg[0]`)** | Source and destination for tile data. Each iteration reads 32 elements (2 physical rows × 16 elements/row) from DEST, processes them, and writes back. |
| **LREG[0]** | Implicit working register for SFPLOAD/SFPSTORE target. Holds the current 32-element vector being processed. |
| **`vConstFloatPrgm0`** | Set by `_init_sfpu_reciprocal_<false>()` to `0.3232325...` — quadratic coefficient k0 for reciprocal initial estimate. |
| **`vConstFloatPrgm1`** | Set by `_init_sfpu_reciprocal_<false>()` to `1.4545459...` — quadratic coefficient k1 for reciprocal initial estimate. |
| **`vConstFloatPrgm2`** | Set by `_init_sfpu_reciprocal_<false>()` to `2.1212124...` — quadratic coefficient k2 for reciprocal initial estimate. |
| **`vConst1`** | Built-in constant 1.0f — used in `_sfpu_exp_` Horner evaluation and `_sfpu_reciprocal_` Newton-Raphson. |
| **`vConst0p8373`** | Built-in constant 0.8373 — used in `_sfpu_exp_` Horner polynomial. |
| **`vConstNeg1`** | Built-in constant -1.0f — used in `_sfpu_reciprocal_` for `setman(vConstNeg1, ...)`. |
| **`slope` parameter** | Runtime `uint32_t` argument passed from the LLK dispatch layer, reinterpreted as float via `Converter::as_float()`. Represents the ELU alpha parameter. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()` (defined in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`).

For ELU (and most unary SFPU operations that don't match the special-cased `SfpuType` values), only `ADDR_MOD_7` is configured:

```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

All address increments are zero because DEST address progression is handled **explicitly** by:
1. **Within a face**: `sfpi::dst_reg++` in the SFPU kernel loop (8 iterations per face, each advancing 1 sfpi row = 2 physical DEST rows)
2. **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` in `_llk_math_eltwise_unary_sfpu_params_` (called twice between each face, advancing 8+8=16 physical DEST rows = 1 face)

This configuration is identical across Wormhole B0 and Blackhole hardware generations.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h`
   **Reason**: Core SFPU kernel for ELU — the primary subject of this analysis.
   **Key Findings**: ELU uses `_calculate_exponential_piecewise_` for the negative branch with a single `slope` (alpha) parameter. For x >= 0, the value passes through unchanged. Uses SFPI abstractions throughout.

2. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_elu.h`
   **Reason**: Checked for hardware-specific differences.
   **Key Findings**: Identical to the Wormhole version.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_calculate_exponential_piecewise_`, `_sfpu_exp_`, `_calculate_exponential_approx_`, and `_init_exponential_` — all called by ELU's kernel and init.
   **Key Findings**: The non-approximate path uses `_sfpu_exp_` (Horner polynomial + repeated squaring) followed by `_sfpu_reciprocal_<2>` for negative inputs. The init for non-approximate, non-fast mode sets up reciprocal polynomial coefficients via `_init_sfpu_reciprocal_<false>()`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains `_sfpu_reciprocal_<2>` and `_init_sfpu_reciprocal_` used by the non-approximate exponential path.
   **Key Findings**: Reciprocal uses a quadratic initial estimate followed by 2 Newton-Raphson iterations for float32 precision (≤1 ULP). The init loads three polynomial coefficients into programmable constant registers.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Contains `Converter::as_float()` used to reinterpret the `uint32_t slope` parameter as a float.
   **Key Findings**: Simple union-based bit reinterpretation from `uint32_t` to `float`.

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked for ELU dispatch wiring (`get_op_approx_mode`, `get_op_init_and_func`, `get_compute_kernel_path`).
   **Key Findings**: `get_op_approx_mode` returns `false` for all ops (default case). No ELU-specific case in `get_op_init_and_func_default` or `get_op_init_and_func_parameterized`. `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` for ELU (default case).

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Confirmed `UnaryOpType::ELU` enum exists.
   **Key Findings**: ELU is listed at line 51 of the enum.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: The shared parameters dispatch function used by all unary SFPU operations.
   **Key Findings**: For `VectorMode::RC`, iterates over 4 faces, calling the SFPU function once per face, with `SETRWC(CR_D, 8)` ×2 between faces.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains `_llk_math_eltwise_unary_sfpu_init_` and `eltwise_unary_sfpu_configure_addrmod`.
   **Key Findings**: Init configures `ADDR_MOD_7` with all increments = 0. ELU doesn't match any special-cased `SfpuType` for additional address mode configuration.

10. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h`
    **Reason**: Verified how `_calculate_elu_` and `_init_elu_` are called in the tt_llk test infrastructure.
    **Key Findings**: Test calls `_init_elu_<APPROX_MODE>()` then `_calculate_elu_<APPROX_MODE, ITERATIONS>(1)` with slope=1 (alpha=1.0).
