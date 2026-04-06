## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the cube root (`cbrt`) operation.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile_init(); cbrt_tile(0);`

**Note on dispatch wiring**: In this worktree, the `CBRT` enum value exists in `UnaryOpType` and the SFPU kernel implementation is fully present, but the dispatch is not wired in `get_op_init_and_func_default()` within `unary_op_utils.cpp`. The `SfpuType::cbrt` enum value is also absent from `llk_sfpu_types.h`. The API header (`cbrt.h`), LLK dispatch files, and core SFPU kernel are all fully implemented. This analysis covers the SFPU kernel implementation as-is.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (JIT-generated constexpr bool from `math_approx_mode`) | `cbrt_tile<APPROX, DST_ACCUM_MODE>(idst)` in `cbrt.h` -- APPROX is the JIT-generated global |
| Effective SFPU path | `APPROXIMATION_MODE=false` always (since `math_approx_mode=false`). The kernel does **not** branch on `APPROXIMATION_MODE` -- it is unused in the function body. The only branch is on `is_fp32_dest_acc_en`, which is controlled by `DST_ACCUM_MODE` (from `fp32_dest_acc_en` in ComputeConfig). | `calculate_cube_root` has `if constexpr (is_fp32_dest_acc_en)` branch only |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` (identical on both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` (identical on both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared params dispatch) |

### Call Chain
1. **`cbrt_tile(idst)`** (API Header: `cbrt.h`) -- wraps the call inside the `MATH()` macro, calling `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en>(dst_index)`** (LLK Dispatch: `llk_math_eltwise_unary_sfpu_cbrt.h`) -- calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` passing `sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>` as the callable, along with `dst_index` and `vector_mode=VectorMode::RC`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu_func, dst_index, vector_mode)`** (Parameters Dispatch: `llk_math_eltwise_unary_sfpu_params.h`) -- sets up DEST write address, configures address mode base, stalls for SFPU readiness, then loops over 4 faces (for `VectorMode::RC`), calling `sfpu_func()` once per face, advancing the DEST face address between faces via `SETRWC`/`inc_dst_addr`.
4. **`calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`** (Core SFPU: `ckernel_sfpu_cbrt.h`) -- executes the Moroz et al. magic-constant cube root algorithm, processing `ITERATIONS=8` sfpi rows per face invocation.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (Face 0 through Face 3), covering the full 32x32 = 1024 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_cube_root()` once per face. Each invocation processes 8 sfpi rows (ITERATIONS=8), which corresponds to one full 16x16 face.
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (the SFPU kernel uses `dst_reg++` internally for per-row advancement). Between faces, `TTI_SETRWC` advances by 8 twice (= 16 physical rows = 1 face). On Blackhole, the same `ADDR_MOD_7` configuration applies, but face advancement uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

This kernel uses **Style A: SFPI-based** abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::abs`, `sfpi::reinterpret`, `sfpi::setsgn`, `sfpi::addexp`, `sfpi::int32_to_float`, `sfpi::float_to_fp16b`).

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h

// This is a modified version of "Fast Calculation of Cube and Inverse Cube
// Roots Using a Magic Constant and Its Implementation on Microcontrollers" by
// Moroz et al. <https://doi.org/10.3390/en14041058>

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=<from ComputeConfig>, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3 / 256 = approx -0.001302083; loaded via SFPLOADI

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // pre-computed constant loaded via SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row
        sfpi::vFloat x = sfpi::abs(a); // SFPABS: x = |a|, preserve sign for later

        // Implementation notes, see the original file for more details
        //
        // Compute initial approximation via magic constant method:
        // Reinterpret float bits as integer, compute 0x548c2b4b - i/3 in float
        // domain, then shift left 8 to recover the integer result.

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0);
        // reinterpret: type-cast vFloat -> vInt (no instruction, same register)
        // int32_to_float: SFPCAST with INT32_TO_FP32_RNE mode -- converts integer bit pattern to float

        f = f * negative_third_256 + magic;
        // SFPMAD: f = f * negative_third_256 + magic (fused multiply-add)

        // Now, left-shift by 8 to restore integer result.

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8);
        // reinterpret vFloat -> vInt: no instruction, same register
        // << 8: SFPSHFT with immediate 8 (logical left shift)
        // reinterpret vInt -> vFloat: no instruction, same register

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: two Newton-Raphson refinement steps for higher precision

            sfpi::vFloat c = (x * y) * (y * y); // 3x SFPMAD: c = x*y*y^2
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0);
            // 3x SFPMAD: polynomial refinement using programmable constants

            sfpi::vFloat d = x * (y * y); // 2x SFPMAD: d = x*y^2
            c = d * y + sfpi::vConstNeg1; // SFPMAD: c = d*y + (-1.0)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8);
            // SFPDIVP2 with ADD mode: adds 8 to exponent, restoring -1/3 from -1/3/256
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // SFPMAD: t = c*(-1/3) + 1.0
            d = sfpi::setsgn(d, a); // SFPSETSGN: copy sign of original input a onto d
            y = d * (t * t); // 2x SFPMAD: y = d * t^2 (final result)

            sfpi::dst_reg[0] = y; // SFPSTORE: write result back to DEST
        } else {
            // FP16B path: single Newton-Raphson refinement step

            sfpi::vFloat d = x * (y * y); // 2x SFPMAD: d = x*y^2
            sfpi::vFloat c = d * y; // SFPMAD: c = d*y = x*y^3
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0;
            // 3x SFPMAD: polynomial refinement using programmable constants
            d = sfpi::setsgn(d, a); // SFPSETSGN: copy sign of original input a onto d
            y = d * (t * t); // 2x SFPMAD: y = d * t^2 (final result)

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
            // float_to_fp16b: SFP_STOCH_RND with FP32_TO_FP16B mode, rounding=NearestEven
            // reinterpret: type-cast vUInt -> vFloat (no instruction)
            // SFPSTORE: write rounded fp16b result back to DEST
        }
        sfpi::dst_reg++; // advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    // Load polynomial refinement coefficients into programmable constant registers
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;   // SFPCONFIG: ~1.75293... (Horner coefficient a0)
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;   // SFPCONFIG: ~-1.25095... (Horner coefficient a1)
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;   // SFPCONFIG: ~0.50945... (Horner coefficient a2)
}
```

### SFPU Instructions Used

| SFPU Instruction | SFPI Abstraction | Description | Usage in CBRT |
|-----------------|------------------|-------------|---------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from DEST row into LREG | Load input value `a` from current DEST position |
| **SFPSTORE** | `dst_reg[0] = ...` (write) | Store LREG value back to DEST row | Write final cube root result to DEST |
| **SFPABS** | `sfpi::abs(a)` | Absolute value (clear sign bit) | Compute `x = |a|` to work with magnitude |
| **SFPCAST** | `sfpi::int32_to_float(v, 0)` | Convert integer to FP32 (RNE rounding) | Reinterpret float bits as integer, then convert to float for arithmetic on the bit pattern |
| **SFPMAD** | `vFloat * vFloat`, `vFloat * vFloat + vFloat` | Fused multiply-add: `VD = VA * VB + VC` | All floating-point multiplications and additions: initial approximation, Newton-Raphson refinement steps, polynomial evaluation |
| **SFPSHFT** | `vInt << 8` | Logical left shift by immediate | Shift result left by 8 bits to undo the /256 scaling used in the magic constant method |
| **SFPSETSGN** | `sfpi::setsgn(d, a)` | Copy sign bit from one value to another | Restore the original sign of the input onto the magnitude result (cube root preserves sign) |
| **SFPDIVP2** | `sfpi::addexp(v, 8)` | Add immediate to exponent field | Restore `-1/3` from `-1/3/256` by adding 8 to the exponent (FP32 path only) |
| **SFP_STOCH_RND** | `sfpi::float_to_fp16b(y, 0)` | Stochastic rounding: FP32 to FP16B | Round the FP32 result to bfloat16 precision before storing (FP16B path only) |
| **SFPLOADI** | `sfpi::vFloat var = constant` | Load 16-bit immediate to LREG | Load pre-computed constants (`negative_third_256`, `magic`) into LREGs at loop start |
| **SFPCONFIG** | `vConstFloatPrgm{0,1,2} = ...` | Configure programmable constant registers | Set polynomial refinement coefficients in `cube_root_init()` |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG0-LREG3** (LREGS1 bank) | General-purpose scratch registers used by the compiler for intermediate values (`a`, `x`, `f`, `y`, `c`, `d`, `t`, `negative_third_256`, `magic`). The SFPI compiler allocates these automatically. |
| **DEST rows** (via `dst_reg`) | Input source and output destination. Each iteration reads from and writes to the current DEST row pair (32 elements). `dst_reg++` advances to the next sfpi row. |
| **Programmable Constant 0** (`vConstFloatPrgm0`) | Horner coefficient a0 = `0x1.c09806p0f` (approximately 1.75293). Set in `cube_root_init()` via SFPCONFIG. |
| **Programmable Constant 1** (`vConstFloatPrgm1`) | Horner coefficient a1 = `-0x1.403e6cp0f` (approximately -1.25095). Set in `cube_root_init()` via SFPCONFIG. |
| **Programmable Constant 2** (`vConstFloatPrgm2`) | Horner coefficient a2 = `0x1.04cdb2p-1f` (approximately 0.50945). Set in `cube_root_init()` via SFPCONFIG. |
| **Fixed Constant `vConstNeg1`** | Pre-defined constant -1.0 (FP32 path only). Used in the second Newton-Raphson refinement step. |
| **Fixed Constant `vConst1`** | Pre-defined constant 1.0 (FP32 path only). Used in the second Newton-Raphson refinement step. |

### Address Mode Configuration

The CBRT operation uses `SfpuType::cbrt` for its `eltwise_unary_sfpu_configure_addrmod()` call. Since `cbrt` does not match any of the special-cased `SfpuType` values (topk_local_sort, typecast, unary_max/min, reciprocal), only the default `ADDR_MOD_7` is configured.

**Wormhole B0 and Blackhole (identical configuration):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Standard SFPU address mode -- no auto-increment on DEST. The SFPU kernel manages DEST advancement internally via `dst_reg++` (which generates SETRWC instructions). |

The `dest.incr = 0` configuration means hardware does not auto-increment the DEST address between SFPU instructions. Instead, the `dst_reg++` abstraction in the SFPI loop explicitly advances the DEST read/write pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements) after each iteration. Between faces, the params dispatch advances by calling `TTI_SETRWC` twice with an increment of 8 (total 16 physical rows = one 16x16 face).

This configuration is identical across Wormhole B0 and Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Consulted to determine `get_op_approx_mode()`, `get_compute_kernel_path()`, and `get_op_init_and_func_default()` for the CBRT operation.
   **Key Findings**: CBRT falls through to `default: return false` for approx mode, falls through to `default: return "eltwise_sfpu.cpp"` for compute kernel path, and is NOT present in `get_op_init_and_func_default()` (would throw TT_THROW at runtime).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header defining `cbrt_tile()` and `cbrt_tile_init()` compute API functions.
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`. `cbrt_tile_init()` calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU implementation.
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, VectorMode::RC)`. WH and BH implementations are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU kernel implementation containing the cube root algorithm.
   **Key Findings**: Implements the Moroz et al. magic-constant method for fast cube root approximation with Newton-Raphson refinement. Has two code paths: FP32 (two refinement steps + SFPDIVP2 for exponent restoration) and FP16B (one refinement step + SFP_STOCH_RND for format conversion). The APPROXIMATION_MODE template parameter is unused -- only is_fp32_dest_acc_en affects the code path.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that handles face iteration and DEST address progression.
   **Key Findings**: For VectorMode::RC, loops over 4 faces, calling the SFPU function once per face. WH version uses explicit TTI_SETRWC instructions for face advancement; BH version uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains `eltwise_unary_sfpu_configure_addrmod()` and `_llk_math_eltwise_unary_sfpu_init_()` which configure ADDR_MOD and initialize SFPU state.
   **Key Findings**: CBRT uses only `ADDR_MOD_7` with all increments = 0 (no special address mode needed). Identical on WH and BH.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Consulted for SFPI abstraction-to-instruction mappings (`abs`, `int32_to_float`, `setsgn`, `addexp`, `float_to_fp16b`, `reinterpret`).
   **Key Findings**: `abs()` maps to `SFPABS`, `int32_to_float()` maps to `SFPCAST`, `setsgn(v, sgn)` with vector args maps to `SFPSETSGN` (vector variant), `addexp()` maps to `SFPDIVP2` with ADD mode, `float_to_fp16b()` maps to `SFP_STOCH_RND` with FP32_TO_FP16B mode.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware reference for tile geometry, DEST layout, addressing modes, and instruction semantics.
   **Key Findings**: Confirmed ITERATIONS=8 per face, stride-2 addressing model, SFPMAD as the universal float add/multiply instruction, and SFPCONFIG for programmable constant register updates.
