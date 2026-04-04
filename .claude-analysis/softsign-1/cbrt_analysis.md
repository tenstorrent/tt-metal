## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized version) | `get_op_init_and_func_default()` -- `cbrt_tile_init()` / `cbrt_tile({idst})` with no explicit template argument; the API header uses compile-time `APPROX` define |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en` resolved from `DST_ACCUM_MODE` | In `calculate_cube_root`, the `if constexpr (is_fp32_dest_acc_en)` branch at line 48 selects between the FP32 path (extra Newton-Raphson iteration) and the FP16B path (single refinement + stochastic round to fp16b) |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` (identical for both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` (identical for both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`cbrt_tile(idst)`** (API header `cbrt.h`) calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` inside `MATH((...))`.
2. **`llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>`** (LLK dispatch `llk_math_eltwise_unary_sfpu_cbrt.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch `llk_math_eltwise_unary_sfpu_params.h`) sets DEST write address, configures address mode, stalls SFPU, then iterates over 4 faces in `VectorMode::RC` mode, calling the SFPU functor once per face with `SETRWC`/`inc_dst_addr` between faces.
4. **`calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS=8>()`** (core SFPU `ckernel_sfpu_cbrt.h`) executes the cube root computation on 8 sfpi iterations per face invocation, processing all 256 elements of that face.

Additionally, **`cbrt_tile_init()`** calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. This initializes the SFPU config register, configures `ADDR_MOD_7`, resets counters, and programs three constant registers via `cube_root_init`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed, covering the full 32x32 = 1024 elements.
- **Operation invocation**: In RC mode, the params dispatch loops `for (face = 0; face < 4; face++)`, calling the SFPU functor (`calculate_cube_root`) once per face. The functor internally loops for `ITERATIONS=8` sfpi rows per face. Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` is called (Blackhole) or two `TTI_SETRWC` with stride 8 are issued (Wormhole) to advance the DEST write pointer past the face.
- **DEST address progression**: On Wormhole, `ADDR_MOD_7` is configured with `.dest = {.incr = 0}`, meaning no auto-increment from the address mode -- the SFPU kernel handles its own DEST addressing via `dst_reg++` (advancing 1 sfpi row = 2 physical DEST rows = 32 elements per iteration). Between faces, `TTI_SETRWC` with stride 8 is issued twice (2 x 8 = 16 physical DEST rows = one face). On Blackhole, the same `ADDR_MOD_7` with zero increment is used, and `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice. Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC/inc_dst_addr between faces).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::abs`, `sfpi::reinterpret`, etc.), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h

// Implementation notes, see the original file for more details
// This is a modified version of "Fast Calculation of Cube and Inverse Cube
// Roots Using a Magic Constant and Its Implementation on Microcontrollers" by
// Moroz et al. <https://doi.org/10.3390/en14041058>

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en from DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3 / 256 = approx -0.001302, loaded via SFPLOADI

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // ~13931142.0f, loaded via SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD: read 32 elements from DEST
        sfpi::vFloat x = sfpi::abs(a);     // SFPABS: absolute value, preserving sign in 'a' for later

        // Reinterpret float bits as integer, convert to float for arithmetic
        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: int32->fp32 (RNE mode)

        f = f * negative_third_256 + magic; // SFPMAD or SFPMUL+SFPADD: compute initial estimate index

        // Left-shift the lower bits by 8 to reconstruct the integer result
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: shift left 8

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: two Newton-Raphson refinement steps
            sfpi::vFloat c = (x * y) * (y * y);  // SFPMUL x2: c = x * y^3
            // Horner evaluation: y *= c*(Prgm2*c + Prgm1) + Prgm0
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // SFPMUL/SFPADD/SFPMAD chain

            // Second Newton-Raphson step for extra precision
            sfpi::vFloat d = x * (y * y);                            // SFPMUL x2: d = x * y^2
            c = d * y + sfpi::vConstNeg1;                            // SFPMAD: c = d*y - 1
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: multiply exponent by adding 8 -> -1/3
            sfpi::vFloat t = c * negative_third + sfpi::vConst1;     // SFPMAD: t = 1 - c/3
            d = sfpi::setsgn(d, a);                                  // SFPSETSGN: restore original sign from input
            y = d * (t * t);                                         // SFPMUL x2: final result

            sfpi::dst_reg[0] = y; // SFPSTORE: write result back to DEST
        } else {
            // FP16B path: single Newton-Raphson refinement + round to fp16b
            sfpi::vFloat d = x * (y * y);   // SFPMUL x2: d = x * y^2
            sfpi::vFloat c = d * y;          // SFPMUL: c = x * y^3
            // Horner evaluation: t = c*(Prgm2*c + Prgm1) + Prgm0
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // SFPMUL/SFPADD/SFPMAD chain
            d = sfpi::setsgn(d, a);          // SFPSETSGN: restore original sign from input
            y = d * (t * t);                 // SFPMUL x2: final result

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND + SFPSTORE: round to fp16b then store
        }
        sfpi::dst_reg++; // advance to next sfpi row (2 physical DEST rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() { // APPROXIMATION_MODE=false
    // Program constant registers with Horner polynomial coefficients
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;   // SFPCONFIG: ~1.7523 (degree-0 coefficient)
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;   // SFPCONFIG: ~-1.2510 (degree-1 coefficient)
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;   // SFPCONFIG: ~0.5094 (degree-2 coefficient)
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| **SFPLOAD** | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST position into an LREG for processing |
| **SFPSTORE** | `sfpi::dst_reg[0] = ...` (write) | Store computed cube root result back to DEST |
| **SFPABS** | `sfpi::abs(a)` | Compute absolute value of input, so cube root is computed on positive values (sign restored later) |
| **SFPCAST** | `sfpi::int32_to_float(v, 0)` | Cast the reinterpreted float bits (as int32) to fp32, used to manipulate the IEEE 754 bit representation arithmetically |
| **SFPMUL** | `vFloat * vFloat` | Floating-point multiply; used extensively for Newton-Raphson refinement products like `x * y^2`, `d * y`, `t * t` |
| **SFPADD** | `vFloat + vFloat` | Floating-point add; used in Horner polynomial evaluation and Newton iteration terms |
| **SFPMAD** | compiler-fused `a * b + c` | The compiler may fuse consecutive SFPMUL+SFPADD into SFPMAD; patterns like `f * neg_third_256 + magic` and `c * neg_third + vConst1` are natural MAD candidates |
| **SFPSHFT** | `vInt << 8` | Left-shift by 8 bits to reconstruct the integer cube root estimate from the mantissa bits of the floating-point intermediate |
| **SFPDIVP2** | `sfpi::addexp(v, 8)` | Add 8 to the exponent field, effectively multiplying by 2^8; used to convert `-1/(3*256)` back to `-1/3` in the FP32 path |
| **SFPSETSGN** | `sfpi::setsgn(d, a)` | Copy the sign of the original input `a` onto the computed magnitude `d`, since cbrt preserves sign (cbrt(-x) = -cbrt(x)) |
| **SFP_STOCH_RND** | `sfpi::float_to_fp16b(y, 0)` | Round fp32 result to fp16b format with stochastic rounding (rounding mode 0); used only in the non-FP32-accumulation path |
| **SFPLOADI** | `sfpi::vFloat var = constant` | Load 32-bit float immediate into an LREG (emitted as two SFPLOADI for upper and lower 16 bits, or SFPMAD with constant regs) |
| **SFPCONFIG** | `sfpi::vConstFloatPrgmN = value` | Program the SFPU constant registers with polynomial coefficients during `cube_root_init()` |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | Working registers for intermediate values (`a`, `x`, `f`, `y`, `c`, `d`, `t`, `negative_third_256`, `magic`). The compiler allocates these dynamically. With up to ~10 live vFloat variables in the FP32 path, register pressure is high and the compiler spills/reloads as needed. |
| **LREG4-LREG7** | Additional working registers available to the compiler for register allocation. |
| **DEST** | Source and destination for tile data. Each `dst_reg[0]` read loads from the current DEST row; each `dst_reg[0] = ...` write stores back. `dst_reg++` advances the DEST pointer by 1 sfpi row (2 physical rows). |
| **Programmable Constant 0 (vConstFloatPrgm0)** | Set to `0x1.c09806p0f` (~1.7523) by `cube_root_init()`. Used as degree-0 coefficient in the Horner polynomial. |
| **Programmable Constant 1 (vConstFloatPrgm1)** | Set to `-0x1.403e6cp0f` (~-1.2510) by `cube_root_init()`. Used as degree-1 coefficient in the Horner polynomial. |
| **Programmable Constant 2 (vConstFloatPrgm2)** | Set to `0x1.04cdb2p-1f` (~0.5094) by `cube_root_init()`. Used as degree-2 coefficient in the Horner polynomial. |
| **Fixed Constant: vConstNeg1** | Hardware constant `-1.0`. Used in the FP32 path for `c = d * y + vConstNeg1` (computing `d*y - 1`). |
| **Fixed Constant: vConst1** | Hardware constant `1.0`. Used in the FP32 path for `t = c * negative_third + vConst1` (computing `1 - c/3`). |

### Address Mode Configuration

The SFPU address mode for `cbrt` is configured via `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()`, which does not match any special-case `if constexpr` branch. Therefore, only the default `ADDR_MOD_7` is set:

**Wormhole B0:**
```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

**Blackhole:**
```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

Both architectures use the identical configuration: `ADDR_MOD_7` with zero increments for srca, srcb, and dest. The SFPU kernel manages its own DEST addressing entirely through `dst_reg++` (which advances the SFPU DEST pointer by 1 sfpi row = 2 physical rows per iteration), with `SETRWC`/`inc_dst_addr` between faces handled by the params dispatch layer.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for CBRT
   **Key Findings**: CBRT uses `eltwise_sfpu.cpp`, dispatches to `cbrt_tile_init()`/`cbrt_tile(idst)`, approx mode is `false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header defining `cbrt_tile()` and `cbrt_tile_init()` functions
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`, passing both approx mode and FP32 accumulation mode as template arguments

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Identical for both architectures. Init calls `llk_math_eltwise_unary_sfpu_init` with `cube_root_init` callback. Tile function calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_cube_root` functor, ITERATIONS=8.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU kernel implementing the cube root algorithm
   **Key Findings**: Uses Moroz et al. magic constant method (0x548c2b4b) for initial estimate, then Newton-Raphson refinement. Two code paths: FP32 (two refinement steps) and FP16B (one refinement + stochastic round). Uses SFPI abstractions throughout.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that manages DEST addressing and face iteration
   **Key Findings**: Wormhole uses `TTI_SETRWC` for face advancement, Blackhole uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`. Both process 4 faces in RC mode.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init function
   **Key Findings**: `ADDR_MOD_7` is set with zero increments (srca=0, srcb=0, dest=0). No special-case addr_mod for `SfpuType::cbrt`.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI abstractions to compiler builtins (and thereby to SFPU instructions)
   **Key Findings**: `abs` -> `SFPABS`, `int32_to_float` -> `SFPCAST`, `addexp` -> `SFPDIVP2`, `setsgn` -> `SFPSETSGN`, `float_to_fp16b` -> `SFP_STOCH_RND`, `reinterpret` -> no-op (type cast only)

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Map vFloat arithmetic operators to SFPU instructions
   **Key Findings**: `vFloat + vFloat` -> `SFPADD` (via `__builtin_rvtt_sfpadd`), `vFloat * vFloat` -> `SFPMUL` (via `__builtin_rvtt_sfpmul`). Compiler may fuse into `SFPMAD`.

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware reference for tile geometry, DEST layout, stride-2 model, and instruction semantics
   **Key Findings**: Confirmed ITERATIONS=8 per face, stride-2 addressing, SFPMAD as the fused multiply-add instruction, and instruction latency/timing rules.

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Verify the include guard mechanism for CBRT
    **Key Findings**: `SFPU_OP_CBRT_INCLUDE` gates inclusion of `api/compute/eltwise_unary/cbrt.h`
