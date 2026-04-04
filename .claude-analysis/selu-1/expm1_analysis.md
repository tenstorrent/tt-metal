## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the `expm1` operation, which computes `exp(x) - 1`.

### Unary Dispatch Summary
- **UnaryOpType**: `EXPM1`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `expm1_tile_init<false>(); expm1_tile<false>(0);`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(EXPM1)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | `expm1_tile<bool approx = false>` and `expm1_tile_init<bool approx = false>` in `compute_kernel_api.h` — the default template argument is `false` |
| Effective SFPU path | Non-approximate: `_sfpu_exp_(setsgn(val, 0))` + `sfpu_reciprocal(out)` for negative inputs, then subtract 1.0 | `if constexpr (APPROXIMATION_MODE)` branch in `calculate_exponential_body_improved` — the `else` branch is taken |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 435–446: `expm1_tile<approx>()` and `expm1_tile_init<approx>()`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_expm1.h` (defines `llk_math_eltwise_unary_sfpu_expm1` and `llk_math_eltwise_unary_sfpu_expm1_init`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h` (defines `calculate_expm1` and `expm1_init`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (defines `_llk_math_eltwise_unary_sfpu_params_` — generic per-face dispatch with `VectorMode::RC`) |

> **Note**: The LLK dispatch and Core SFPU files (`llk_math_eltwise_unary_sfpu_expm1.h`, `ckernel_sfpu_expm1.h`) reside in the `llk_sfpu/` directory alongside other per-operation kernel files. They are identical between Wormhole and Blackhole architectures. These files are not present in the `tt_llk` submodule — they are generated or placed by a separate code generation tool (`tt_ops_code_gen`).

### Call Chain

1. **API → LLK dispatch**: `expm1_tile<false>(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_expm1<false, DST_ACCUM_MODE>(idst)))`.
2. **LLK dispatch → Params dispatch**: `llk_math_eltwise_unary_sfpu_expm1<false>(dst_index, VectorMode::RC)` calls `llk_math_eltwise_unary_sfpu_params<false>(ckernel::sfpu::calculate_expm1<false>, dst_index, VectorMode::RC)`.
3. **Params dispatch → Core SFPU**: `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces in `VectorMode::RC`, calling `calculate_expm1<false>()` per face, with `SETRWC` advancing the DEST address between faces.
4. **Core SFPU → Exp helper**: `calculate_expm1<false>()` loops 8 iterations per face, loading each element from `dst_reg[0]`, passing it to `calculate_exponential_body_improved<false>()`, subtracting 1.0, and storing the result back.
5. **Exp helper → Primitive functions**: `calculate_exponential_body_improved<false>(val)` computes `sfpu_exp(setsgn(val, 0))` via `_sfpu_exp_()` (Horner series), then conditionally applies `sfpu_reciprocal(out)` via `_sfpu_reciprocal_<3>()` (Newton-Raphson) when `val < 0`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) — all 4 faces of the tile are processed, covering all 1024 elements.
- **Operation invocation**: For RC mode, the params dispatch loops over 4 faces. Each iteration calls `calculate_expm1<false>()` which internally loops 8 times (`ITERATIONS=8`), processing 32 elements per iteration (2 physical DEST rows × 16 elements/row via stride-2 addressing).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `ADDR_MOD_7` is configured with all increments set to 0 (srca=0, srcb=0, dest=0) — the `expm1` SfpuType does not match any of the specialized address mode cases (`topk_local_sort`, `typecast`, etc.). Blackhole uses the same `ADDR_MOD_7` configuration. The DEST address advances within each face via explicit `dst_reg++` in the kernel loop, and between faces via `TTI_SETRWC` calls in the params dispatch.

### Annotated SFPU Kernel Source

The expm1 kernel is SFPI-based (uses `vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_expm1() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];                                             // SFPLOAD from DEST
        v = calculate_exponential_body_improved<APPROXIMATION_MODE>(v);    // compute exp(v)
        dst_reg[0] = v - 1.0f;                                            // SFPMAD(v, 1.0, -1.0) then SFPSTORE
        dst_reg++;                                                         // advance DEST by 1 sfpi row (= 2 physical rows)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

The `calculate_exponential_body_improved` helper (from `ckernel_sfpu_exp.h` in the arch-specific `llk_sfpu/` directory) is the actual exp computation:

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_exponential_body_improved(vFloat val) { // APPROXIMATION_MODE=false
    vFloat out;
    if constexpr (APPROXIMATION_MODE) {
        // === APPROXIMATE PATH (not taken for expm1 default) ===
        v_if(val >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;                                              // Saturate to +inf for large inputs
        }
        v_elseif(val < -42) { out = 0.0f; }                            // Saturate to 0 for very negative inputs
        v_else {
            // Bit-manipulation approximation: x * (1/ln2) + bias, reinterpret as float
            vFloat vConstLn2Recip = vConstFloatPrgm0;                   // 1.442695 (1/ln2)
            vFloat c23_73 = vConstFloatPrgm1;                           // p_exp::C23_73 = 0x4340
            vInt adj_exp = vConstIntPrgm2;                              // p_exp::ADJ_EXP = 0xBD3F
            val = val * vConstLn2Recip + c23_73;                        // SFPMAD: convert to 7.3 FxP format
            vInt val_short = adj_exp + reinterpret<vInt>(val);          // Remove exponent of 7, bias mantissa to 127
            val_short <<= 10 - p_exp::FRAC_BITS;                       // SHL by 7 to move integer bits to exponent field
            out = reinterpret<vFloat>(val_short);
        }
        v_endif;
    } else {
        // === NON-APPROXIMATE PATH (taken for expm1 default) ===
        out = sfpu_exp(setsgn(val, 0));                                 // exp(|val|) via Horner series (_sfpu_exp_)
        v_if(val < 0) { out = sfpu_reciprocal(out); }                  // 1/exp(|val|) = exp(-|val|) for negative inputs
        v_endif;
    }
    return out;
}
```

The `_sfpu_exp_` function (from `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h`) performs the core exponential computation:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // Extract exponent; if >= 0, scale input to [-1, 1) range
    sfpi::vInt exp = exexp(val);                                     // SFPEXEXP: extract biased exponent
    v_if (exp >= 0)
    {
        val = setexp(val, 126);                                      // SFPSETEXP: set exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Horner-form polynomial: exp(x) ≈ x * (x * 0.8373 + 0.863281) + 1
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281); // SFPMAD
    val              = val * tmp + sfpi::vConst1;                                // SFPMAD

    // Square-and-multiply for the integer part of the exponent
    v_if (exp >= 0)
    {
        val = val * val;                                              // SFPMAD (square)
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;                                            // SFPIADD: decrement exponent counter
            v_and(exp >= 0);                                          // narrow predication
            val = val * val;                                          // SFPMAD (conditional square)
        }
    }
    v_endif;

    return val;
}
```

The `_sfpu_reciprocal_` function (from `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_recip.h`) is called by `sfpu_reciprocal` with `max_iter=3`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) // max_iter=3 when called from sfpu_reciprocal
{
    // Scale input to [-2, -1) range and make negative
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;              // SFPMAD

    // Compute scale factor via bit inversion of exponent
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);                                 // SFPNOT

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;                                                // SFPMAD

    // Clear mantissa from scale
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);           // SFPSETMAN

    // Newton-Raphson iteration 1: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;                                            // SFPMAD

    // Adjust scale exponent: scale *= 0.5 (subtracts 1 from exponent)
    scale *= 0.5f;                                                                               // SFPMAD

    // Newton-Raphson: y = y + y*t
    y = y + y * t;                                                                               // SFPMAD

    if constexpr (max_iter > 1)
    {
        // Newton-Raphson iteration 2
        t = sfpi::vConst1 + negative_x * y;                                                     // SFPMAD
        y = y + y * t;                                                                           // SFPMAD
    }

    // (max_iter > 2 would add a third iteration, but the code only has max_iter <= 2 branches)

    // Apply scaling and restore sign
    y = y * scale;                                                                               // SFPMAD
    y = sfpi::setsgn(y, in);                                                                     // SFPSETSGN

    return y;
}
```

The `expm1_init` function configures the programmable constant registers:

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h

template <bool APPROXIMATION_MODE>
void expm1_init() { // APPROXIMATION_MODE=false
    if constexpr (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f;              // 1/ln(2) for base conversion
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);  // 0x4340: 7.3 FxP bias
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP); // 0xBD3F: exponent adjustment
    } else {
        vConstFloatPrgm0 = 1.442695f;              // 1/ln(2) — used by _sfpu_exp_ Horner series
        vConstFloatPrgm1 = 2.0f;                   // Horner coefficient
        vConstFloatPrgm2 = 0.863281f;              // Horner coefficient
    }
}
```

> **Implementation note**: The non-approximate path of `calculate_exponential_body_improved` calls `sfpu_reciprocal()` → `_sfpu_reciprocal_<3>()` for negative inputs. The reciprocal function uses `vConstFloatPrgm{0,1,2}` as polynomial coefficients for its quadratic initial estimate (expected values: 0.323, 1.455, 2.121 from `_init_sfpu_reciprocal_`). However, `expm1_init<false>` sets these registers to `{1.442695, 2.0, 0.863281}` — values for the exp Horner series, not the reciprocal polynomial. The `_sfpu_exp_` function itself does NOT read from `vConstFloatPrgm` registers (it uses built-in constants `vConst0p8373` and literal `s2vFloat16b(0.863281)`). This means the non-approximate path may produce reduced accuracy for negative inputs due to incorrect reciprocal initialization. Compare with the standard `exp` operation's `_init_exponential_<false>()` which correctly calls `_init_sfpu_reciprocal_<false>()` to set the proper reciprocal polynomial coefficients.

### SFPU Instructions Used

| Instruction / Intrinsic | Description | Used By |
|--------------------------|-------------|---------|
| `SFPLOAD` / `dst_reg[0]` read | Load 32 elements from DEST register into LREG | `calculate_expm1` — load input element |
| `SFPSTORE` / `dst_reg[0]` write | Store 32 elements from LREG back to DEST register | `calculate_expm1` — store `exp(x) - 1` result |
| `SFPMAD` / `vFloat + vFloat`, `vFloat * vFloat` | Fused multiply-add (a * b + c); also emitted for float addition (a * 1.0 + b) and multiplication (a * b + 0.0) | `_sfpu_exp_` Horner polynomial, `_sfpu_reciprocal_` Newton-Raphson, `v - 1.0f` subtraction |
| `SFPEXEXP` / `exexp(val)` | Extract exponent field from float | `_sfpu_exp_` — extract exponent for range reduction |
| `SFPSETEXP` / `setexp(val, 126)` | Set exponent field of float to a constant | `_sfpu_exp_` — scale input to [-1, 1) range |
| `SFPSETSGN` / `setsgn(val, 0)` | Set sign bit of float | `calculate_exponential_body_improved` — force positive; `_sfpu_reciprocal_` — restore sign |
| `SFPSETMAN` / `setman(a, b)` | Set mantissa of float | `_sfpu_reciprocal_` — scale input to [-2, -1) and clear mantissa from scale |
| `SFPNOT` / `~vUInt` | Bitwise NOT | `_sfpu_reciprocal_` — compute scale factor by inverting exponent bits |
| `SFPIADD` / `vInt - 1` | Integer addition/subtraction | `_sfpu_exp_` — decrement exponent counter in square-and-multiply loop |
| `SFPSETCC` / `v_if`, `v_elseif`, `v_and` | Set condition codes for predicated execution | Control flow in `_sfpu_exp_` (exp >= 0), `calculate_exponential_body_improved` (val < 0, val >= 89) |
| `SFPLOADI` / `s2vFloat16b(...)` | Load immediate constant into LREG | Literal constants (0.863281, 0.5, 1.0, etc.) |
| `SFPSHFT` / `val_short <<= N` | Shift left (integer) | Approximate path: shift integer bits to exponent field |
| `SETRWC` | Set read/write counters for DEST face advance | Params dispatch: advance between tile faces |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register** | Input tile data (read) and output tile data (write). Each `dst_reg[0]` access reads/writes 32 elements (2 physical rows × 16 elements/row). |
| **LREG0–LREG3** | Working registers used by SFPI abstractions (`vFloat`, `vInt`). The compiler allocates these for intermediate values: `v`, `out`, `exp`, `tmp`, `negative_x`, `y`, `scale`, `t` in the reciprocal chain. |
| **vConstFloatPrgm0** | Non-approx: `1.442695` (1/ln2). Approx: `1.442695` (1/ln2). |
| **vConstFloatPrgm1** | Non-approx: `2.0`. Approx: `s2vFloat16b(p_exp::C23_73)` = `s2vFloat16b(0x4340)`. |
| **vConstFloatPrgm2** | Non-approx: `0.863281`. Approx: `s2vFloat16b(p_exp::ADJ_EXP)` = `s2vFloat16b(0xBD3F)`. |
| **vConst0p8373** | Built-in SFPI constant (≈0.8373). Used by `_sfpu_exp_` Horner polynomial. |
| **vConst1** | Built-in SFPI constant (1.0). Used by `_sfpu_exp_` and `_sfpu_reciprocal_`. |
| **vConstNeg1** | Built-in SFPI constant (-1.0). Used by `_sfpu_reciprocal_` to create negative scaled input. |

### Address Mode Configuration

The `expm1` operation uses `SfpuType::expm1` during initialization, which triggers `eltwise_unary_sfpu_configure_addrmod<SfpuType::expm1>()`. Since `expm1` does not match any specialized case in the address mode configuration function, only the default `ADDR_MOD_7` is configured:

**ADDR_MOD_7 (both Wormhole and Blackhole):**
```
srca.incr = 0
srcb.incr = 0
dest.incr = 0
```

This is the standard configuration for unary SFPU operations. The DEST address increment is 0 because `calculate_expm1` manually advances the DEST pointer via `dst_reg++` in its loop (1 sfpi row = 2 physical DEST rows per iteration), and the params dispatch uses explicit `TTI_SETRWC` instructions to advance between faces.

The address mode configuration is identical between Wormhole and Blackhole for this operation.

## Local Knowledge Sources

### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN defines
   **Key Findings**: `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default), `get_op_approx_mode` returns `false` (default). EXPM1 init/func strings are generated by `tt_ops_code_gen`.

2. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 435–446)
   **Reason**: Trace the API-level tile functions for expm1
   **Key Findings**: `expm1_tile<bool approx = false>(uint32_t idst)` calls `llk_math_eltwise_unary_sfpu_expm1<approx, DST_ACCUM_MODE>(idst)`. Init: `expm1_tile_init<bool approx = false>()`.

3. **File**: `/localdev/adjordjevic/work/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_expm1.h`
   **Reason**: LLK dispatch layer — bridges API to core SFPU function
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(ckernel::sfpu::calculate_expm1<APPROXIMATE>, ...)` with `VectorMode::RC`.

4. **File**: `/localdev/adjordjevic/work/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h`
   **Reason**: Core SFPU kernel — the actual expm1 implementation
   **Key Findings**: Simple 8-iteration loop: load from DEST, call `calculate_exponential_body_improved`, subtract 1.0, store back.

5. **File**: `/localdev/adjordjevic/work/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `calculate_exponential_body_improved` and `sfpu_exp` wrapper
   **Key Findings**: Non-approx path computes `exp(|x|)` via `_sfpu_exp_`, then `1/exp(|x|)` via `sfpu_reciprocal` for negative inputs.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Shared SFPU exp implementation (`_sfpu_exp_`)
   **Key Findings**: Uses Horner series with square-and-multiply for range reduction. Built-in constants (`vConst0p8373`, `s2vFloat16b(0.863281)`) — does NOT use `vConstFloatPrgm` registers.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Reciprocal implementation used by non-approx exp path
   **Key Findings**: Quadratic initial estimate + Newton-Raphson iterations. Uses `vConstFloatPrgm{0,1,2}` for polynomial coefficients.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic params dispatch function for unary SFPU operations
   **Key Findings**: `VectorMode::RC` processes all 4 faces with `SETRWC` between faces. WH version uses `TTI_SETRWC` directly; BH version calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::expm1>` only sets `ADDR_MOD_7` with zero increments (expm1 doesn't match any specialized case).

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h`
    **Reason**: `p_exp` constant definitions
    **Key Findings**: `p_exp::FRAC_BITS = 3`, `p_exp::C23_73 = 0x4340`, `p_exp::ADJ_EXP = 0xBD3F`.
