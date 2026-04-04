## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSIGN`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `softsign_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSIGN)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` returns `softsign_tile_init()` / `softsign_tile(idst)` with no parameterized template arguments; `APPROX` defaults to the `math_approx_mode` value (`false`) |
| Effective SFPU path | On Wormhole: `_sfpu_reciprocal_<2>` uses 2 Newton-Raphson iterations with a quadratic initial estimate. `softsign_init` calls `_init_sfpu_reciprocal_<false>()` which programs `vConstFloatPrgm0/1/2` with polynomial coefficients. On Blackhole: `_sfpu_reciprocal_<2>` uses `SFPARECIP` hardware instruction + 2 Newton-Raphson iterations. `softsign_init` calls `_init_sfpu_reciprocal_<false>()` which programs `vConstFloatPrgm0 = 2.0f`. | Wormhole: `if constexpr (max_iter > 1)` branch in `ckernel_sfpu_recip.h:64`. Blackhole: `if constexpr (max_iter > 1)` branch in `ckernel_sfpu_recip.h:35` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h` (identical on both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h` (identical on both architectures; calls into architecture-specific `sfpu/ckernel_sfpu_recip.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. `softsign_tile(idst)` (API header) calls `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)` via the `MATH()` macro.
2. `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)` (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_softsign<APPROX, 8>, dst_index, VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (params dispatch) sets up DEST addressing, stalls for SFPU availability, then calls `calculate_softsign<false, 8>()` once per face (4 times total for `VectorMode::RC`), advancing the DEST face pointer between calls.
4. `calculate_softsign<false, 8>()` (core SFPU) iterates 8 times per face, and for each iteration: loads from DEST, computes `abs(x) + 1.0`, calls `_sfpu_reciprocal_<2>()`, multiplies `x * reciprocal`, and stores back to DEST.
5. `_sfpu_reciprocal_<2>(denom)` (shared primitive in `sfpu/ckernel_sfpu_recip.h`) computes the reciprocal with 2 Newton-Raphson iterations. The implementation differs between Wormhole and Blackhole (see Annotated SFPU Kernel Source below).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls `calculate_softsign()` 4 times in a loop (once per face). Each invocation internally iterates 8 times (ITERATIONS=8) processing 32 elements per iteration (2 physical DEST rows x 16 elements/row).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr<8>` between faces). Only `ADDR_MOD_7` is configured (dest increment = 0), meaning hardware auto-increment is not used; software `dst_reg++` handles intra-face progression and `SETRWC` handles inter-face advancement.

### Annotated SFPU Kernel Source

The core SFPU kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`). Style A applies.

The softsign kernel is identical on both Wormhole and Blackhole. However, the `_sfpu_reciprocal_<2>` helper it calls has architecture-specific implementations. Both are included below.

#### `calculate_softsign` -- Core kernel (identical on WH and BH)

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softsign() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];             // SFPLOAD: load 32 elements from current DEST rows

        // Compute denominator: 1 + |x|
        sfpi::vFloat denom = sfpi::abs(v) + sfpi::vConst1; // SFPABS + SFPMAD(|x| * 1.0 + 1.0)

        // Compute reciprocal of denominator: 1 / (1 + |x|)
        sfpi::vFloat recip = _sfpu_reciprocal_<2>(denom);  // See architecture-specific implementation below

        // Result: x * (1 / (1 + |x|))
        sfpi::dst_reg[0] = v * recip;                  // SFPMAD(v * recip + 0.0) then SFPSTORE
        sfpi::dst_reg++;                                // Advance DEST pointer by 1 sfpi row (= 2 physical rows)
    }
}
```

#### `_sfpu_reciprocal_<2>` -- Wormhole B0 implementation

This version uses a software quadratic initial estimate followed by 2 Newton-Raphson iterations.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) // max_iter=2
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: scale input to [-2,-1), preserving sign/exponent of -1.0

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD: prgm0 * neg_x + prgm1

    // Implementation notes, see the original file for more details
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT: bitwise complement for exponent inversion (255-in.Exp)

    // Continue with quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;      // SFPMAD: y * neg_x + prgm2

    // Scale factor: set mantissa to zero
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN: clear mantissa bits

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;  // SFPMAD: neg_x * y + 1.0

    // Scale factor adjustment: scale = scale*0.5
    scale *= 0.5f;                                      // SFPMUL: scale * 0.5

    // Continue Newton-Raphson: y = y + y*t
    y = y + y * t;                                      // SFPMAD: y * t + y

    // Second Newton-Raphson iteration (max_iter > 1): t = 1.0 - x*y; y = y + y*t
    t = sfpi::vConst1 + negative_x * y;                // SFPMAD: neg_x * y + 1.0
    y = y + y * t;                                      // SFPMAD: y * t + y

    // Apply scaling factor, and set sign to match input
    y = y * scale;                                      // SFPMAD: y * scale + 0.0
    y = sfpi::setsgn(y, in);                            // SFPSETSGN: copy sign from original input to result

    return y;
}
```

#### `_sfpu_reciprocal_<2>` -- Blackhole implementation

This version uses the hardware `SFPARECIP` instruction for the initial approximation, followed by 2 Newton-Raphson iterations with NaN guarding.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x) // max_iter=2
{
    // SFPARECIP: hardware approximate reciprocal; returns +/-0 for +/-inf, +/-inf for +/-0
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Implementation notes, see the original file for more details
    sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;   // SFPMAD: x * y - 2.0 (negated error term)

    // max_iter > 1 branch:
    sfpi::vFloat y1 = y * -t - sfpi::vConst0;          // SFPMAD with sign inversion: y * (-t) - 0.0 = y * (-t)
    // If t=NaN, then t>=0. This check consumes the SFPNOP slot of the preceding SFPMAD.
    v_if (t < 0)                                        // SFPSETCC + SFPENCC: guard against NaN (0/0 or inf/inf case)
    {
        t = x * y1 - sfpi::vConstFloatPrgm0;           // SFPMAD: second NR error term
        y = y1 * -t - sfpi::vConst0;                   // SFPMAD: second NR correction
    }
    v_endif;                                            // SFPENCC: restore all-lanes-active

    return y;
}
```

#### `_init_sfpu_reciprocal_` -- Initialization (architecture-specific)

**Wormhole B0**: Programs the polynomial coefficients for the quadratic initial estimate.
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0: x^2 coefficient
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1: x coefficient (negated in use)
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2: constant term
}
```

**Blackhole**: Programs the constant `2.0` used in the Newton-Raphson error term.
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 2.0f;    // Used as subtrahend in NR: t = x*y - 2.0
    }
}
```

### SFPU Instructions Used

The following table lists all SFPU instructions emitted by the softsign kernel and its reciprocal helper. Instructions are grouped by the function that emits them.

#### Instructions from `calculate_softsign` (both architectures)

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST rows into LREG |
| `SFPABS` | `sfpi::abs(v)` | Compute absolute value of the input (clears sign bit) |
| `SFPMAD` | `abs(v) + vConst1` | Fused multiply-add: `|x| * 1.0 + 1.0` to compute denominator |
| `SFPMAD` | `v * recip` | Fused multiply-add: `x * reciprocal + 0.0` to compute final result |
| `SFPSTORE` | `dst_reg[0] = ...` (write) | Store 32 elements back to DEST rows |

#### Instructions from `_sfpu_reciprocal_<2>` -- Wormhole B0

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPSETMAN` | `sfpi::setman(vConstNeg1, ...)` | Combines sign/exponent of -1.0 with mantissa of input, scaling to [-2,-1) |
| `SFPMAD` | `vConstFloatPrgm1 + vConstFloatPrgm0 * neg_x` | First step of quadratic initial estimate |
| `SFPNOT` | `~reinterpret<vUInt>(in)` | Bitwise complement to compute 255-exponent for scale factor |
| `SFPMAD` | `vConstFloatPrgm2 + y * neg_x` | Second step of quadratic initial estimate |
| `SFPSETMAN` | `sfpi::setman(..., 0)` | Clear mantissa of scale factor |
| `SFPMAD` | `vConst1 + neg_x * y` | Newton-Raphson: compute error term `t = 1.0 - x*y` (first iteration) |
| `SFPMUL` | `scale *= 0.5f` | Halve the scale factor (adjusts exponent bias) |
| `SFPMAD` | `y + y * t` | Newton-Raphson: apply correction `y = y + y*t` (first iteration) |
| `SFPMAD` | `vConst1 + neg_x * y` | Newton-Raphson: compute error term (second iteration) |
| `SFPMAD` | `y + y * t` | Newton-Raphson: apply correction (second iteration) |
| `SFPMAD` | `y * scale` | Apply scaling factor to final reciprocal result |
| `SFPSETSGN` | `sfpi::setsgn(y, in)` | Copy sign bit from original input to result |

#### Instructions from `_sfpu_reciprocal_<2>` -- Blackhole

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPARECIP` | `sfpi::approx_recip(x)` | Hardware approximate reciprocal (Blackhole-specific) |
| `SFPMAD` | `x * y - vConstFloatPrgm0` | Compute negated NR error term: `t = x*y - 2.0` |
| `SFPMAD` | `y * (-t) - vConst0` | First NR correction: `y1 = y * (-t)` (with sign inversion via InstrMod) |
| `SFPSETCC` | `v_if (t < 0)` | Set CC for NaN guard (t < 0 means no NaN) |
| `SFPMAD` | `x * y1 - vConstFloatPrgm0` | Second NR error term (CC-guarded) |
| `SFPMAD` | `y1 * (-t) - vConst0` | Second NR correction (CC-guarded) |
| `SFPENCC` | `v_endif` | Restore all-lanes-active CC state |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output: each iteration reads 2 physical rows (32 elements) from DEST via `SFPLOAD`, computes softsign, and writes back via `SFPSTORE` |
| **LREG0-3** | General-purpose working registers used by the SFPI compiler for intermediate values (`v`, `denom`, `recip`, `y`, `t`, `scale`, etc.). Exact allocation is compiler-determined. |
| **vConstFloatPrgm0** | Wormhole: `0.3232325` (k0 polynomial coefficient). Blackhole: `2.0` (Newton-Raphson constant). Programmed by `softsign_init()` -> `_init_sfpu_reciprocal_<false>()`. |
| **vConstFloatPrgm1** | Wormhole only: `1.4545459` (k1 polynomial coefficient). |
| **vConstFloatPrgm2** | Wormhole only: `2.121212` (k2 polynomial coefficient). |
| **vConst1** | Fixed constant `1.0`, used in denominator computation (`1 + |x|`) and Newton-Raphson error terms. |
| **vConstNeg1** | Wormhole only: Fixed constant `-1.0`, used as source for `setman` to scale input to [-2,-1). |
| **vConst0** | Blackhole only: Fixed constant `0.0`, used as addend in SFPMAD for multiply-only operations. |

### Address Mode Configuration

The softsign operation uses `SfpuType::softsign`, which does not match any special-cased `if constexpr` branch in `eltwise_unary_sfpu_configure_addrmod`. Therefore, only the default address mode is configured:

| Address Mode | Configuration | Hardware |
|-------------|---------------|----------|
| `ADDR_MOD_7` | `srca.incr=0, srcb.incr=0, dest.incr=0` | Both Wormhole and Blackhole |

The dest increment is 0 because the SFPU uses software-managed addressing:
- **Within a face**: `dst_reg++` in the SFPI code advances the SFPU read/write pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration.
- **Between faces**: The params dispatch function calls `SETRWC` (Wormhole: two `TTI_SETRWC` with increment 8 each) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole: two `inc_dst_addr<8>()` calls) to advance by 16 physical DEST rows = 1 face.

Both architectures use the same `ADDR_MOD_7` configuration for this operation. No `ADDR_MOD_6` is configured since `SfpuType::softsign` is not in the reciprocal/typecast/min/max special-case list.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SOFTSIGN.
   **Key Findings**: SOFTSIGN uses `eltwise_sfpu.cpp`, expands to `softsign_tile_init()` / `softsign_tile(idst)`, `get_op_approx_mode` returns `false` (default case).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`
   **Reason**: Identify the API-level function signatures and the first hop in the call chain.
   **Key Findings**: `softsign_tile(idst)` calls `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)`. `softsign_tile_init()` calls `llk_math_eltwise_unary_sfpu_softsign_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`
   **Reason**: Trace the LLK dispatch layer to find the core SFPU function and params dispatch.
   **Key Findings**: Dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_softsign<APPROXIMATE, 8>` and `VectorMode::RC`. Init calls `softsign_init<APPROXIMATE>`.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
   **Reason**: Read the core SFPU kernel implementation.
   **Key Findings**: Both architectures have identical kernel code. Computes `x / (1 + |x|)` using `sfpi::abs`, `sfpi::vConst1`, `_sfpu_reciprocal_<2>`, and multiply. Init programs reciprocal constants via `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Read the Wormhole implementation of `_sfpu_reciprocal_<2>` and `_init_sfpu_reciprocal_`.
   **Key Findings**: Uses quadratic polynomial initial estimate (`y = k2 - k1*x + k0*x^2`) with mantissa/exponent manipulation (`setman`, `SFPNOT`) + 2 Newton-Raphson iterations. Init programs `vConstFloatPrgm0/1/2` with Sollya-optimized polynomial coefficients.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Read the Blackhole implementation of `_sfpu_reciprocal_<2>` and `_init_sfpu_reciprocal_`.
   **Key Findings**: Uses hardware `SFPARECIP` instruction + 2 NR iterations with NaN guarding via `v_if (t < 0)`. Init programs `vConstFloatPrgm0 = 2.0f`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch pattern (face iteration, DEST address progression).
   **Key Findings**: `VectorMode::RC` iterates over all 4 faces, calling the SFPU function once per face, with `SETRWC`/`inc_dst_addr` between faces.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration (`eltwise_unary_sfpu_configure_addrmod`).
   **Key Findings**: `SfpuType::softsign` only configures `ADDR_MOD_7` with all increments = 0. No `ADDR_MOD_6` is configured for this operation.

9. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Verify SFPI intrinsic-to-instruction mappings (`abs` -> `SFPABS`, `setman` -> `SFPSETMAN`, `setsgn` -> `SFPSETSGN`, `approx_recip` -> `SFPARECIP`).
   **Key Findings**: All mappings confirmed. `approx_recip` is Blackhole-only (`__riscv_xtttensixbh` guard).

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU addressing model, instruction semantics, and tile geometry.
    **Key Findings**: Used for DEST stride-2 model, ITERATIONS=8 per face, SFPMAD semantics, and programmable constant register layout.
