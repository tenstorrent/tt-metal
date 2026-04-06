## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile(0)`

**Note**: In this worktree (`gen-frac-v4`), the `CBRT` dispatch is not wired into `unary_op_utils.cpp` (the `get_op_init_and_func_default` switch does not have a `CBRT` case, and `SfpuType::cbrt` is absent from `llk_sfpu_types.h`). However, all SFPU kernel files exist intact. The analysis below documents the full SFPU implementation as it would operate when the dispatch path is complete.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (default) | `cbrt_tile_init()` and `cbrt_tile(idst)` use `APPROX` which comes from the compute kernel's `APPROX` define (set from `math_approx_mode`); no additional parameterization |
| Effective SFPU path | `APPROXIMATION_MODE=false` | The `calculate_cube_root` template receives `APPROXIMATION_MODE=false`, but the kernel does not branch on `APPROXIMATION_MODE` -- it is unused in the implementation. The only `if constexpr` branch is on `is_fp32_dest_acc_en`. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** calls `cbrt_tile(idst)` (the `SFPU_OP_CHAIN_0` macro expansion).
2. **API header** (`cbrt.h`) wraps this as `MATH((llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)))`, dispatching to the math thread.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_cbrt.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls for SFPU readiness, then iterates over 4 faces (for `VectorMode::RC`), calling `calculate_cube_root()` once per face with `SETRWC`/`_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces.
5. **Core SFPU function** (`ckernel_sfpu_cbrt.h`) performs the actual cube root computation using SFPI abstractions.

For initialization: `cbrt_tile_init()` calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`, which invokes `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. This runs the standard `_llk_math_eltwise_unary_sfpu_init_<SfpuType::cbrt>()` (address mode configuration + counter reset) and then calls `cube_root_init()` to load polynomial coefficients into programmable constant registers.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls `calculate_cube_root()` once per face (4 times total for RC mode). Each call processes 8 sfpi iterations (ITERATIONS=8 default), covering one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_face_addr` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` to advance by 8+8=16 sfpi rows between faces. On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.
- **Address mode**: `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` (all zero increments). Since `SfpuType::cbrt` does not match any special-case `if constexpr` in `eltwise_unary_sfpu_configure_addrmod`, only the default ADDR_MOD_7 is set. This is the same on both Wormhole and Blackhole.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, etc.) -- Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h
// (Wormhole and Blackhole implementations are identical)

// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en depends on DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3/256 ~= -0.001302083 -> SFPLOADI

    // Magic constant: 0x548c2b4b / 256 + 2^23 = 5540907.293 + 8388608.0 = 13929515.293
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // -> SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD: read 32 elements from current DEST position
        sfpi::vFloat x = sfpi::abs(a); // SFPABS: take absolute value (sign handled later via setsgn)

        // Implementation notes, see the original file for more details

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: reinterpret float bits as int, convert int32->fp32 with RNE rounding

        f = f * negative_third_256 + magic; // SFPMAD: f = f * (-1/3/256) + magic_constant

        // Left-shift by 8 to restore the integer result from the scaled fp32 domain
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: integer left-shift by 8 bits

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: Householder-order refinement (two stages)
            sfpi::vFloat c = (x * y) * (y * y); // 2x SFPMAD: c = x*y^3
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // 3x SFPMAD: y = y * P(c) where P is the Moroz polynomial

            // Halley refinement step
            sfpi::vFloat d = x * (y * y); // 2x SFPMAD: d = x*y^2
            c = d * y + sfpi::vConstNeg1; // SFPMAD: c = d*y - 1 = x*y^3 - 1 (residual)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: multiply by 2^8, restoring -1/3
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // SFPMAD: t = 1 - c/3 (Halley correction factor)
            d = sfpi::setsgn(d, a); // SFPSETSGN: restore original sign from input a
            y = d * (t * t); // 2x SFPMAD: y = d * t^2 (final refined result)

            sfpi::dst_reg[0] = y; // SFPSTORE: write back to DEST
        } else {
            // FP16B path: single Householder refinement
            sfpi::vFloat d = x * (y * y); // 2x SFPMAD: d = x*y^2
            sfpi::vFloat c = d * y; // SFPMAD: c = d*y = x*y^3
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // 3x SFPMAD: t = P(c) Moroz polynomial
            d = sfpi::setsgn(d, a); // SFPSETSGN: restore original sign from input a
            y = d * (t * t); // 2x SFPMAD: y = d * t^2

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND + SFPSTORE: convert to fp16b with RNE rounding, then store
        }
        sfpi::dst_reg++; // advance to next sfpi row (next 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() { // APPROXIMATION_MODE=false (unused, no branching on it)
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f; // SFPCONFIG -> Prog Const 3 (CREG 12): ~1.7523197 (Moroz polynomial coefficient a0)
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f; // SFPCONFIG -> Prog Const 4 (CREG 13): ~-1.2509525 (Moroz polynomial coefficient a1)
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f; // SFPCONFIG -> Prog Const 5 (CREG 14): ~0.5093818 (Moroz polynomial coefficient a2)
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST row into LREG for processing |
| `SFPABS` | `sfpi::abs(a)` | Compute absolute value of input (sign bit cleared); the kernel operates on |x| and restores sign at the end |
| `SFPCAST` | `sfpi::int32_to_float(v, 0)` | Reinterpret the FP32 bit pattern of |x| as an integer, then convert that integer back to FP32 (with RNE rounding). This extracts the IEEE 754 bit representation as a numerical value for the magic-constant trick |
| `SFPMAD` | `*`, `+` on vFloat | Fused multiply-add (a * b + c). Used extensively for polynomial evaluation and arithmetic. All vFloat additions compile to SFPMAD (a * 1.0 + b) since there is no dedicated float add instruction |
| `SFPSHFT` | `vInt << 8` | Integer left-shift by 8 bits. Restores the scaled magic-constant result to the correct integer range after the fp32-domain computation |
| `SFPSETSGN` | `sfpi::setsgn(d, a)` | Copy the sign bit from the original input `a` onto the magnitude result `d`. This restores the correct sign since the kernel computes cbrt(|x|) and then applies sgn(x) |
| `SFPDIVP2` | `sfpi::addexp(v, 8)` | Add 8 to the exponent field, effectively multiplying by 2^8 = 256. Used in FP32 path to recover -1/3 from -1/3/256 |
| `SFP_STOCH_RND` | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 result to FP16B (bfloat16) with RNE (Round to Nearest Even) rounding. Only used in the non-FP32 DEST path |
| `SFPSTORE` | `dst_reg[0] = ...` (write) | Store the computed result back to the current DEST row |
| `SFPLOADI` | vFloat literal assignment | Load 16-bit immediate values to construct the `negative_third_256` and `magic` constants at the start of the function |
| `SFPCONFIG` | `vConstFloatPrgm{0,1,2} = ...` | Write polynomial coefficients to programmable constant registers during `cube_root_init()` |

### SFPU Register Usage

| Register / Resource | Usage |
|---------------------|-------|
| **DEST rows** | Input/output: each iteration reads 2 physical DEST rows (32 elements) via `dst_reg[0]`, processes them, and writes the result back to the same location |
| **LREGs (general purpose)** | Used implicitly by the SFPI compiler for intermediate values: `a`, `x`, `f`, `y`, `c`, `d`, `t`, `negative_third_256`, `magic`. The compiler allocates these across LREG0-LREG7 as needed. The kernel has up to ~6 live vFloat variables simultaneously, fitting within the 8 available LREGs |
| **Prog Const 2 (CREG 11)** | `vConstNeg1` = -1.0 (hardware reset value). Used in FP32 path for Halley refinement: `c = d*y + vConstNeg1` computes the residual x*y^3 - 1 |
| **Prog Const 3 (CREG 12)** | `vConstFloatPrgm0` = 0x1.c09806p0f (~1.7523197). Moroz polynomial constant term a0 |
| **Prog Const 4 (CREG 13)** | `vConstFloatPrgm1` = -0x1.403e6cp0f (~-1.2509525). Moroz polynomial linear coefficient a1 |
| **Prog Const 5 (CREG 14)** | `vConstFloatPrgm2` = 0x1.04cdb2p-1f (~0.5093818). Moroz polynomial quadratic coefficient a2 |
| **Fixed Const 2 (CREG 10)** | `vConst1` = 1.0 (hardware fixed). Used in FP32 path for Halley correction: `t = c * negative_third + vConst1` |

### Address Mode Configuration

The SFPU address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` during initialization.

Since `SfpuType::cbrt` does not match any `if constexpr` special case in the address mode configuration function, only the default `ADDR_MOD_7` is set:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is identical for both Wormhole and Blackhole. The zero-increment address mode means the SFPU does not auto-increment the DEST address between instructions -- address progression is managed entirely by `dst_reg++` (which emits `SFPINCRWC` in SFPI) within the kernel loop, and by `SETRWC`/`inc_dst_face_addr` between faces in the params dispatch.

No additional ADDR_MOD registers (ADDR_MOD_6, etc.) are configured for this operation, unlike operations such as `topk_local_sort` or `typecast` that require special address modes.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU implementation for Wormhole
   **Key Findings**: Complete SFPI-based cube root kernel using Moroz et al. magic-constant method with Householder polynomial refinement. Two code paths: FP32 (with extra Halley step) and FP16B (with float_to_fp16b conversion). WH and BH implementations are identical.

2. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU implementation for Blackhole
   **Key Findings**: Identical to the Wormhole version.

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header exposing `cbrt_tile()` and `cbrt_tile_init()`
   **Key Findings**: Standard unary API pattern: `cbrt_tile(idst)` dispatches to `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` on the MATH thread.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU function
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function for Wormhole
   **Key Findings**: Standard VectorMode::RC dispatch: iterates over 4 faces, calls the SFPU function once per face, uses `TTI_SETRWC` with increment 8 twice between faces (advancing 16 sfpi rows = 1 face).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function for Blackhole
   **Key Findings**: Same VectorMode::RC dispatch pattern, but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of raw `TTI_SETRWC`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: ADDR_MOD configuration and init infrastructure
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` only sets the default `ADDR_MOD_7` with all-zero increments. No special-case handling for cbrt.

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library functions mapping abstractions to hardware instructions
   **Key Findings**: `int32_to_float` -> SFPCAST, `abs` -> SFPABS, `setsgn` -> SFPSETSGN, `addexp` -> SFPDIVP2, `float_to_fp16b` -> SFP_STOCH_RND, `reinterpret` -> no-op (type cast only), `<<` on vInt -> SFPSHFT.

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Unary dispatch configuration
   **Key Findings**: CBRT is not wired into `get_op_init_and_func_default()` or `get_op_init_and_func_parameterized()` in this worktree. `get_op_approx_mode()` returns `false` for all ops (default case). `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` for all ops (default case).

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware reference for instruction semantics, register layout, and constant register values
    **Key Findings**: Confirmed Prog Const 2 (CREG 11) has reset value -1.0 (`vConstNeg1`), Fixed Const 2 (CREG 10) is 1.0 (`vConst1`). SFPCAST performs int32-to-fp32 conversion. SFPDIVP2 adds to the exponent field. SFP_STOCH_RND handles float-to-float format conversion.

11. **File**: `runtime/sfpi/include/sfpi_constants.h`
    **Reason**: CREG index definitions and SFP_DESTREG_STRIDE
    **Key Findings**: Confirmed CREG_IDX_NEG_1 = CREG_IDX_PRGM0 = 11, CREG_IDX_1 = 10, CREG_IDX_PRGM1 = 12, CREG_IDX_PRGM2 = 13, CREG_IDX_PRGM3 = 14.
