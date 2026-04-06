## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile_init(); cbrt_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` compile-time define from `math_approx_mode`) | `cbrt_tile_init()` and `cbrt_tile(idst)` in `cbrt.h` use the global `APPROX` define, which resolves to `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false`. However, the kernel does not have any `if constexpr (APPROXIMATION_MODE)` branches -- it ignores the approximation mode entirely. The only branching is on `is_fp32_dest_acc_en`, which controls whether an extra Newton-Raphson refinement step is performed and whether the output is converted to FP16B. | `ckernel_sfpu_cbrt.h` lines 48-68: `if constexpr (is_fp32_dest_acc_en)` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

Note: The Wormhole B0 and Blackhole implementations of the core SFPU kernel (`ckernel_sfpu_cbrt.h`) are **identical**.

### Call Chain

1. **`cbrt_tile(idst)`** (API header `cbrt.h`) calls `MATH(llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst))`.
2. **`llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>(dst_index, vector_mode=RC)`** (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(callable, dst_index, vector_mode, ...)`** (params dispatch in tt_llk) sets DEST write address, sets address mode base, stalls for SFPU availability, then calls the SFPU function once per face (4 times for `VectorMode::RC`), advancing DEST address with `TTI_SETRWC` between faces.
4. **`calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`** (core SFPU in `ckernel_sfpu_cbrt.h`) executes the Moroz cube root algorithm for 8 iterations (one face).

For init: **`cbrt_tile_init()`** calls `MATH(llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>())` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. This configures SFPU address modes and then calls `cube_root_init()` to load programmable constants.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed. In RC mode, the params dispatch loops over 4 faces, calling the SFPU function once per face.
- **Operation invocation**: The core function `calculate_cube_root` is called 4 times (once per face). Each invocation runs 8 iterations (`ITERATIONS=8`), processing 32 elements per iteration (2 physical DEST rows x 16 elements/row), totaling 256 elements per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` advances by 16 physical rows between faces). The address mode used is `ADDR_MOD_7` on both Wormhole and Blackhole (configured in `eltwise_unary_sfpu_configure_addrmod` with `dest.incr = 0` -- SFPU manages its own DEST addressing via `dst_reg++`).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::abs`, `sfpi::reinterpret`, etc.), so Style A is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h

// Implementation notes, see the original file for more details
// This is a modified version of "Fast Calculation of Cube and Inverse Cube
// Roots Using a Magic Constant and Its Implementation on Microcontrollers" by
// Moroz et al. <https://doi.org/10.3390/en14041058>

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en depends on DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3/256 ~ -0.001302, loaded via SFPLOADI

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // ~13929515.293, loaded via SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];   // SFPLOAD: load 32 elements from current DEST position
        sfpi::vFloat x = sfpi::abs(a);        // SFPABS: take absolute value (cbrt is odd function, handle sign at end)

        // Implementation notes, see the original file for more details
        // Compute initial approximation: i = 0x548c2b4b - i/3
        // Using FP32 arithmetic + bit extraction trick

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: reinterpret FP32 bits as integer, then cast integer to float (RNE mode)

        f = f * negative_third_256 + magic;  // SFPMAD: f = f * (-1/3/256) + (0x548c2b4b/256 + 2^23)

        // Now, left-shift by 8 to restore integer result.
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: left-shift 8 bits, then reinterpret as float

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: two refinement steps (higher accuracy)
            sfpi::vFloat c = (x * y) * (y * y);  // SFPMAD x2: c = x * y^3
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // SFPMAD x3: first polynomial refinement

            sfpi::vFloat d = x * (y * y);         // SFPMAD: d = x * y^2
            c = d * y + sfpi::vConstNeg1;          // SFPMAD: c = d*y - 1 = x*y^3 - 1 (residual)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: multiply exponent by 2^8, recovering -1/3
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // SFPMAD: t = 1 - (x*y^3-1)/3 (Newton-Raphson correction)
            d = sfpi::setsgn(d, a);                // SFPSETSGN: restore original sign from input a
            y = d * (t * t);                       // SFPMAD x2: y = d * t^2 (apply correction)

            sfpi::dst_reg[0] = y;                  // SFPSTORE: write result back to DEST
        } else {
            // FP16B path: single refinement step
            sfpi::vFloat d = x * (y * y);          // SFPMAD: d = x * y^2
            sfpi::vFloat c = d * y;                // SFPMAD: c = x * y^3
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // SFPMAD x3: polynomial refinement
            d = sfpi::setsgn(d, a);                // SFPSETSGN: restore original sign
            y = d * (t * t);                       // SFPMAD x2: y = d * t^2

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND + SFPSTORE: round to FP16B then store
        }
        sfpi::dst_reg++;                           // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    // Load programmable constants used by the refinement polynomial
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;   // ~1.7523197 -- SFPCONFIG: write to programmable constant register 0
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;   // ~-1.2509525 -- SFPCONFIG: write to programmable constant register 1
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;   // ~0.5093818 -- SFPCONFIG: write to programmable constant register 2
}
```

### Algorithm Overview

The CBRT kernel implements a fast cube root approximation based on the Moroz et al. paper. The algorithm works in three phases:

1. **Initial approximation via magic constant**: The input's IEEE 754 bit pattern is reinterpreted as an integer, then the linear approximation `y0 = reinterpret_as_float(0x548c2b4b - i/3)` is computed. Since the SFPU lacks integer division, this is achieved using FP32 arithmetic (multiply by -1/3/256 plus a magic constant) followed by a left-shift by 8 to undo the scaling. This produces a reasonable first approximation of `cbrt(|x|)`.

2. **Polynomial refinement**: The initial approximation is refined using a polynomial correction derived from Halley's or Householder's iteration. The coefficients `vConstFloatPrgm0` (~1.752), `vConstFloatPrgm1` (~-1.251), and `vConstFloatPrgm2` (~0.509) parameterize this correction polynomial.

3. **Sign restoration and output**: Since `cbrt(-x) = -cbrt(x)`, the kernel computes on `|x|` and restores the original sign at the end using `SFPSETSGN`.

The FP32 path (`is_fp32_dest_acc_en=true`) adds an extra Newton-Raphson-like refinement step for higher accuracy, using `addexp` (SFPDIVP2) to recover the exact `-1/3` constant from the scaled `-1/3/256`. The FP16B path skips this step and rounds the result to bfloat16 via stochastic rounding (`SFP_STOCH_RND`).

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST position into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Store 32 elements from LREG back to current DEST position |
| `SFPABS` | `sfpi::abs(a)` | Compute absolute value (clear sign bit) of FP32 vector |
| `SFPCAST` | `sfpi::int32_to_float(v, 0)` | Cast integer to FP32 (round-to-nearest-even mode, mode=0) |
| `SFPMAD` | `vFloat * vFloat`, `a * b + c` | Fused multiply-add; used for all float multiplications and additions |
| `SFPSHFT` | `vInt << 8` | Logical left-shift by 8 bits (integer operation) |
| `SFPSETSGN` | `sfpi::setsgn(d, a)` | Copy sign bit from source `a` to destination `d` |
| `SFPDIVP2` | `sfpi::addexp(v, 8)` | Add 8 to exponent field (multiply by 2^8), used to recover -1/3 from -1/3/256 |
| `SFP_STOCH_RND` | `sfpi::float_to_fp16b(y, 0)` | Stochastic rounding from FP32 to FP16B format (FP16B path only) |
| `SFPLOADI` | `sfpi::vFloat v = constant` | Load 16-bit immediate to LREG (two instructions for 32-bit FP32 constants) |
| `SFPCONFIG` | `sfpi::vConstFloatPrgmN = ...` | Write programmable constant registers (in `cube_root_init`) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile data is read from and results written back to DEST, advancing 2 physical rows per iteration via `dst_reg++` |
| **LREGs (general purpose)** | Multiple LREGs are used transiently by the SFPI compiler to hold intermediate values: `a` (original input), `x` (absolute value), `f` (integer-as-float), `y` (initial approximation), `c`, `d`, `t` (refinement intermediates). The SFPI compiler manages LREG allocation automatically. |
| **Programmable Constant 0 (vConstFloatPrgm0)** | `0x1.c09806p0f` (~1.7523197) -- polynomial coefficient |
| **Programmable Constant 1 (vConstFloatPrgm1)** | `-0x1.403e6cp0f` (~-1.2509525) -- polynomial coefficient |
| **Programmable Constant 2 (vConstFloatPrgm2)** | `0x1.04cdb2p-1f` (~0.5093818) -- polynomial coefficient |
| **Fixed Constant vConstNeg1** | `-1.0` -- used in FP32 path residual computation (`d*y - 1`) |
| **Fixed Constant vConst1** | `1.0` -- used in FP32 path Newton-Raphson correction (`1 - residual/3`) |

### Address Mode Configuration

The address mode for CBRT is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()`, which sets `ADDR_MOD_7` with all increments at zero:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is the same on both **Wormhole B0** and **Blackhole**. The zero-increment address mode means the hardware does not auto-increment DEST addressing between SFPU instructions -- instead, the SFPU kernel manages DEST addressing explicitly via `dst_reg++` (which advances the SFPU DEST pointer by 1 sfpi row = 2 physical rows per iteration) and `TTI_SETRWC` (which advances between faces in the params dispatch layer).

CBRT does not match any of the special-case `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod` (no `ADDR_MOD_6` is configured), so only `ADDR_MOD_7` is set.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine `get_op_approx_mode()`, `get_compute_kernel_path()`, and `get_block_defines()` behavior for CBRT
   **Key Findings**: CBRT falls through to defaults: `approx_mode=false`, compute kernel=`eltwise_sfpu.cpp`, macro define=`SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: Trace the tile-level API call (`cbrt_tile`, `cbrt_tile_init`) to LLK dispatch
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`, `cbrt_tile_init()` calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: Trace LLK dispatch to core SFPU implementation
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`. Default ITERATIONS=8.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Read core SFPU kernel implementation
   **Key Findings**: Uses Moroz et al. magic constant method. Branches on `is_fp32_dest_acc_en` for FP32 vs FP16B output paths. FP32 path has extra Newton-Raphson refinement. FP16B path uses stochastic rounding. WH and BH implementations identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand parameters dispatch and face iteration pattern
   **Key Findings**: For `VectorMode::RC`, calls SFPU function 4 times (once per face) with `TTI_SETRWC` advancing DEST by 16 physical rows between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration for `SfpuType::cbrt`
   **Key Findings**: CBRT uses default `ADDR_MOD_7` with zero increments. No special address modes configured.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI helper functions to SFPU hardware instructions
   **Key Findings**: `abs()` -> `SFPABS`, `int32_to_float()` -> `SFPCAST`, `addexp()` -> `SFPDIVP2`, `setsgn()` -> `SFPSETSGN`, `float_to_fp16b()` -> `SFP_STOCH_RND`

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand vFloat/vInt operator mappings and constant register definitions
   **Key Findings**: `vFloat * vFloat` -> `SFPMAD`, `vInt << unsigned` -> `SFPSHFT`, `reinterpret<>` is a zero-cost type pun (no instruction emitted), `vConstFloatPrgm{0,1,2}` map to programmable constant registers, `vConst1`/`vConstNeg1` map to fixed constant registers

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU addressing model, instruction semantics, and register layout
   **Key Findings**: SFP_DESTREG_STRIDE=2, ITERATIONS=8 per face, 32 elements per sfpi row, 4 faces per tile = 1024 elements total
