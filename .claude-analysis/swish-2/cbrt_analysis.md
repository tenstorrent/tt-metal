## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the cube root (`cbrt`) operation.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile(0)`

**Note on dispatch mechanism**: `CBRT` is registered via `REGISTER_UNARY_OPERATION(cbrt, CBRT)` in `unary.hpp`, which builds a `UnaryWithParam{UnaryOpType::CBRT}` with no parameters. However, `CBRT` does not appear in the `get_op_init_and_func_default` or `get_op_init_and_func_parameterized` switch statements in `unary_op_utils.cpp`. This means the old `unary_op_utils.cpp` dispatch path will throw for CBRT. The operation must be dispatched through a newer mechanism (e.g., the `unary_ng` path or a direct compute kernel API include). The compute kernel API header at `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` exposes `cbrt_tile()` and `cbrt_tile_init()`, which are available via the `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` macro.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported here.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (compile-time define, defaults to value of `math_approx_mode`) | `cbrt_tile()` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` — `APPROX` is a global compile-time constant derived from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false`. Both fp32 and fp16b branches exist but are selected by `is_fp32_dest_acc_en`, not by approximation mode. The `APPROXIMATION_MODE` template parameter is accepted but **not used** in the current implementation — both branches execute the same algorithm regardless. | The `calculate_cube_root` function never references `APPROXIMATION_MODE` in any `if constexpr` or conditional logic. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** calls `cbrt_tile(idst)` via the `SFPU_OP_CHAIN_0` macro expansion.
2. **API Header** (`cbrt.h`): `cbrt_tile(idst)` wraps `MATH((llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)))`, gating execution to the MATH thread.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_cbrt.h`): `llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>(dst_index, vector_mode)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls until SFPU is ready, then iterates over faces (4 for RC mode), calling `calculate_cube_root()` once per face with `SETRWC`/`inc_dst_addr` between faces.
5. **Core SFPU** (`ckernel_sfpu_cbrt.h`): `calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()` executes the cube root algorithm on 8 sfpi rows (one face of 256 elements).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) — all 4 faces of the tile are processed (32×32 = 1024 elements total).
- **Operation invocation**: The params dispatch calls `calculate_cube_root()` once per face (4 calls total for RC mode). Each invocation processes `ITERATIONS=8` sfpi rows, covering one full 16×16 face (256 elements). The dispatch loop is: `for (int face = 0; face < 4; face++) { sfpu_func(); inc_dst_face_addr(); }`.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the WH-specific `_llk_math_eltwise_unary_sfpu_params_` uses direct `TTI_SETRWC` calls with `p_setrwc::CR_D, 8` (advance by 8 sfpi rows = 16 physical rows = one face) between face invocations. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (advancing 16 physical rows total per face transition).

### Annotated SFPU Kernel Source

The kernel uses **Style A: SFPI-based abstractions** (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::reinterpret`, `sfpi::abs`, `sfpi::setsgn`, etc.). No raw `TT_`/`TTI_` instructions are present.

The WH and BH implementations are **identical** (`ckernel_sfpu_cbrt.h` is the same file in both `wormhole_b0` and `blackhole` ckernels directories).

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h

// Implementation notes, see the original file for more details
// Based on "Fast Calculation of Cube and Inverse Cube Roots Using a Magic
// Constant and Its Implementation on Microcontrollers" by Moroz et al.
// https://doi.org/10.3390/en14041058

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3 / 256 ≈ -0.001302083 — loaded via SFPLOADI into LREG

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // ≈ 13929376.7 — precomputed and loaded via SFPLOADI

#pragma GCC unroll 8 // Hint to fully unroll, one iteration per sfpi row
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations = one 16×16 face
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair
        sfpi::vFloat x = sfpi::abs(a);     // SFPABS: clear sign bit, x = |a|

        // Reinterpret the float bit pattern as integer, convert to fp32 for
        // arithmetic. This lets us do "integer" operations (i/3) using the
        // SFPU's fp32 MAD pipeline.
        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: int32→fp32 (RNE mode)

        f = f * negative_third_256 + magic; // SFPMAD: f = f * (-1/3/256) + (0x548c2b4b/256 + 2^23)

        // Left-shift by 8 to undo the /256 and extract integer result
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: logical left shift 8 bits

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: two-step Halley refinement for higher precision
            sfpi::vFloat c = (x * y) * (y * y);          // SFPMAD chain: c = x * y³
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // Horner polynomial: y *= P(c)

            sfpi::vFloat d = x * (y * y);                // d = x * y² (should be ≈ cbrt(x)² * y² ≈ 1)
            c = d * y + sfpi::vConstNeg1;                 // c = d*y - 1 (error term)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: multiply by 2^8, restoring -1/3
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // t = 1 - c/3 (Halley correction)
            d = sfpi::setsgn(d, a);                       // SFPSETSGN: restore original sign from input a
            y = d * (t * t);                              // Final result: d * t²

            sfpi::dst_reg[0] = y;                         // SFPSTORE: write result back to DEST
        } else {
            // FP16b path: single refinement step (lower precision sufficient)
            sfpi::vFloat d = x * (y * y);                 // d = x * y²
            sfpi::vFloat c = d * y;                        // c = d * y = x * y³
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // Horner polynomial: t = P(c)
            d = sfpi::setsgn(d, a);                        // SFPSETSGN: restore original sign
            y = d * (t * t);                               // Final result: d * t²

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFPSTOCHRND: fp32→fp16b conversion (RNE mode), then SFPSTORE
        }
        sfpi::dst_reg++; // Advance to next sfpi row (+2 physical DEST rows, +32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() { // APPROXIMATION_MODE=false
    // Load Horner polynomial coefficients into programmable constant registers
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;  // ≈ 1.7537820339 — a₀ coefficient
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;  // ≈ -1.2509524822 — a₁ coefficient
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;  // ≈ 0.5093669891 — a₂ coefficient
}
```

### SFPU Instructions Used

| SFPI Abstraction | Underlying SFPU Instruction | Description |
|---|---|---|
| `sfpi::dst_reg[0]` (read) | `SFPLOAD` | Load 32 elements from current DEST row pair into LREG |
| `sfpi::abs(a)` | `SFPABS` | Clear sign bit of float value (take absolute value) |
| `sfpi::int32_to_float(v, 0)` | `SFPCAST` | Reinterpret integer bit pattern as integer, then convert int32 to fp32 using round-to-nearest-even |
| `f * neg + magic` | `SFPMAD` | Fused multiply-add: `a * b + c`. Used extensively for arithmetic (multiplication emitted as `SFPMAD(a, b, 0.0)`; addition emitted as `SFPMAD(a, 1.0, b)`) |
| `sfpi::reinterpret<vInt>(f) << 8` | `SFPSHFT` | Logical left shift by immediate (8 bits), treating float bit pattern as integer |
| `sfpi::setsgn(d, a)` | `SFPSETSGN` (vector variant) | Copy sign bit from `a` to `d`, preserving magnitude of `d` — restores the original sign of the input |
| `sfpi::addexp(v, 8)` | `SFPDIVP2` | Add integer `8` to the exponent field of a float, effectively multiplying by 2⁸ = 256 |
| `sfpi::float_to_fp16b(y, 0)` | `SFPSTOCHRND` | Convert fp32 to fp16b (bfloat16) format with round-to-nearest-even (RNE) mode |
| `sfpi::dst_reg[0] = y` (write) | `SFPSTORE` | Store 32 elements from LREG back to current DEST row pair |
| `sfpi::dst_reg++` | (address counter increment) | Advance the internal sfpi row counter by 1 (= 2 physical DEST rows) |
| `sfpi::vFloat val = literal` | `SFPLOADI` | Load immediate float constant into an LREG |
| `sfpi::vConstFloatPrgm0 = val` | `SFPLOADI` (to CREG) | Load a constant into a programmable constant register (CREG), done during init |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output: each iteration reads 32 elements via `dst_reg[0]` and writes the cube root result back to the same location |
| **LREGs (L0-L7)** | Temporary computation registers. The SFPI compiler maps `vFloat` local variables to LREGs. Key temporaries: `a` (input), `x` (abs), `f` (int-as-float), `y` (initial guess), `c`, `d`, `t` (refinement intermediates). Up to ~6 LREGs may be live simultaneously (compiler manages allocation). |
| **CREG PRGM0** (`vConstFloatPrgm0`) | Horner coefficient a₀ ≈ 1.7537820339 (`0x1.c09806p0f`), loaded in `cube_root_init()` |
| **CREG PRGM1** (`vConstFloatPrgm1`) | Horner coefficient a₁ ≈ -1.2509524822 (`-0x1.403e6cp0f`), loaded in `cube_root_init()` |
| **CREG PRGM2** (`vConstFloatPrgm2`) | Horner coefficient a₂ ≈ 0.5093669891 (`0x1.04cdb2p-1f`), loaded in `cube_root_init()` |
| **CREG NEG1** (`vConstNeg1`) | Built-in constant -1.0, used in fp32 path for error term `c = d*y - 1` |
| **CREG 1** (`vConst1`) | Built-in constant 1.0, used in fp32 path for Halley correction `t = 1 - c/3` |

### Address Mode Configuration

The address mode for the cbrt SFPU operation is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` (called from `_llk_math_eltwise_unary_sfpu_init_<SfpuType::cbrt>()`).

Since `SfpuType::cbrt` does not match any special-cased types in the `if constexpr` branches, only the default `ADDR_MOD_7` is configured:

**Both Wormhole and Blackhole** (identical configuration):
```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

This means the hardware does **not** auto-increment any addressing registers between SFPU instructions. All DEST address progression is handled explicitly:
- **Within a face**: `sfpi::dst_reg++` in the kernel loop advances by 1 sfpi row per iteration
- **Between faces**: The params dispatch calls `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` twice (Blackhole) to advance by one face (16 physical DEST rows = 8 sfpi rows)

No additional `ADDR_MOD_6` is needed because the cbrt kernel does not use any special dual-register or typecast addressing patterns.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Traced dispatch path for CBRT — checked `get_op_approx_mode()`, `get_op_init_and_func_default()`, `get_compute_kernel_path()`, and `get_block_defines()`
   **Key Findings**: CBRT falls through to `default: return false` for approx mode and `default: return "eltwise_sfpu.cpp"` for compute kernel path. CBRT is NOT in the default dispatch switch — it must be dispatched through the compute kernel API include mechanism.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Verified CBRT registration as `REGISTER_UNARY_OPERATION(cbrt, CBRT)` with no parameters
   **Key Findings**: CBRT is a simple non-parameterized unary operation.

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header exposing `cbrt_tile()` and `cbrt_tile_init()` to compute kernels
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` wrapped in `MATH()`. `cbrt_tile_init()` calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU implementation
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`. WH and BH files are identical.

5. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU implementation — the primary subject of this analysis
   **Key Findings**: Uses Moroz et al. magic-constant method (0x548c2b4b) for initial guess via `int32_to_float` + `SFPMAD` + `SFPSHFT`, then Horner polynomial refinement via programmable constants. FP32 path has additional Halley iteration for higher precision. FP16b path has single refinement + `float_to_fp16b` conversion. WH and BH files are identical.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates per-face SFPU execution
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode::RC by iterating over 4 faces, calling the SFPU function once per face, and advancing DEST address by one face stride between calls.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base SFPU init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` only sets `ADDR_MOD_7` with all-zero increments (cbrt is not special-cased). `_llk_math_eltwise_unary_sfpu_init_` calls `_init_sfpu_config_reg()`, then the addr_mod config, then `reset_counters`.

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library function definitions for intrinsics used by the cbrt kernel
   **Key Findings**: `int32_to_float` → `SFPCAST`, `addexp` → `SFPDIVP2`, `setsgn` → `SFPSETSGN`, `abs` → `SFPABS`, `float_to_fp16b` → `SFPSTOCHRND`. The `<<` operator on `vInt` → `SFPSHFT`.

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU architecture, stride-2 addressing, and tile geometry
   **Key Findings**: Confirmed ITERATIONS=8 per face, `dst_reg++` = 2 physical rows = 32 elements, 4 faces × 8 iterations = 32 sfpi iterations = full tile.
