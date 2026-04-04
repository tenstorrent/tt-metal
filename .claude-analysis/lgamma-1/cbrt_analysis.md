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
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` -- non-parameterized: `cbrt_tile_init()` / `cbrt_tile(idst)`, no template parameter in the chain macro. Default template arg `APPROX` comes from compute kernel's `MATH_APPROX_MODE` define, which equals `math_approx_mode` = `false`. |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en` depends on `DST_ACCUM_MODE` | In `calculate_cube_root`, the `APPROXIMATION_MODE` template parameter is not used by any `if constexpr` branch. The `is_fp32_dest_acc_en` parameter controls whether an additional Newton-Raphson refinement step is applied (fp32 path) or whether the result is rounded to fp16b before storing (non-fp32 path). |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`cbrt_tile(idst)`** (API header `cbrt.h`) calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` via the `MATH()` wrapper.
2. **`llk_math_eltwise_unary_sfpu_cbrt<APPROX, fp32_dest_acc_en, ITERATIONS=8>`** (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(sfpu::calculate_cube_root<APPROX, fp32_dest_acc_en, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch in tt_llk) sets up DEST addressing, stalls until SFPU is ready, then loops over 4 faces calling `calculate_cube_root<APPROX, fp32_dest_acc_en, 8>()` once per face with `SETRWC`/`inc_dst_addr` between faces.
4. **`calculate_cube_root<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS=8>()`** (core SFPU implementation in `ckernel_sfpu_cbrt.h`) executes the magic-constant-based cube root algorithm on 8 SFPU iterations per face (256 elements per face).

Additionally, **`cbrt_tile_init()`** calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROX>(sfpu::cube_root_init<APPROX>)` which:
- Calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::cbrt>()` to configure SFPU (init config reg, set `ADDR_MOD_7`, reset counters).
- Calls `cube_root_init<APPROX>()` to program three constant registers via `SFPCONFIG`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed, covering all 1024 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_cube_root()` once per face. Each invocation runs `ITERATIONS=8` loop iterations internally, processing 8 sfpi rows x 32 elements = 256 elements = 1 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` to advance by 8+8=16 physical DEST rows between faces. On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, etc.) -- Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h
// NOTE: WH and BH implementations are identical.

// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=depends on DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3/256 ~= -0.001302; loaded via SFPLOADI

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // ~13929515.293; loaded via SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD: load current element from DEST
        sfpi::vFloat x = sfpi::abs(a);     // SFPABS: take absolute value (cbrt is odd function, sign restored later)

        // Implementation notes, see the original file for more details

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: reinterpret FP32 bits as int, convert to float (round-to-nearest-even)

        f = f * negative_third_256 + magic; // SFPMAD: f = f * (-1/3/256) + (0x548c2b4b/256 + 2^23) -- computes scaled initial guess

        // Now, left-shift by 8 to restore integer result.

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: left-shift by 8 to undo the /256 scaling, then reinterpret as float -- initial cbrt estimate

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 path: two Newton-Raphson-like refinement steps
            sfpi::vFloat c = (x * y) * (y * y);  // SFPMAD/SFPMUL: c = x * y^3 (ideally ~1 if y is perfect cbrt)
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // SFPMAD chain: first Halley-like refinement using programmable constants

            sfpi::vFloat d = x * (y * y);         // SFPMAD: d = x * y^2
            c = d * y + sfpi::vConstNeg1;          // SFPMAD: c = d*y - 1 = x*y^3 - 1 (residual)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: multiply exponent by adding 8 -> recovers -1/3 from -1/3/256
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // SFPMAD: t = 1 - residual/3 (Newton step correction)
            d = sfpi::setsgn(d, a);                // SFPSETSGN: restore original sign from input a
            y = d * (t * t);                       // SFPMAD chain: final refined cbrt = d * t^2

            sfpi::dst_reg[0] = y;                  // SFPSTORE: write result back to DEST
        } else {
            // FP16B path: single refinement step, then round to fp16b
            sfpi::vFloat d = x * (y * y);          // SFPMAD: d = x * y^2
            sfpi::vFloat c = d * y;                 // SFPMAD: c = d * y = x * y^3
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // SFPMAD chain: Halley-like refinement polynomial
            d = sfpi::setsgn(d, a);                // SFPSETSGN: restore original sign
            y = d * (t * t);                       // SFPMAD chain: final refined cbrt = d * t^2

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND + SFPSTORE: round to fp16b (round-to-nearest-even, mode=0) then store
        }
        sfpi::dst_reg++;                           // advance to next sfpi row (2 physical DEST rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f; // SFPCONFIG: ~1.7523 (Halley polynomial coefficient p0)
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f; // SFPCONFIG: ~-1.2510 (Halley polynomial coefficient p1)
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f; // SFPCONFIG: ~0.5094 (Halley polynomial coefficient p2)
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row pair into LREG for processing |
| `SFPSTORE` | `sfpi::dst_reg[0] = y` (write) | Store 32 processed elements from LREG back to DEST row pair |
| `SFPLOADI` | `sfpi::vFloat var = literal` | Load 16-bit immediate constant into LREG (two instructions for 32-bit float literals like `negative_third_256` and `magic`) |
| `SFPABS` | `sfpi::abs(a)` | Take absolute value of float vector (clear sign bit). Used to work with magnitude for the magic constant method, since cbrt is an odd function. |
| `SFPCAST` | `sfpi::int32_to_float(vInt, 0)` | Convert integer (reinterpreted FP32 bit pattern) to float. Mode 0 = round-to-nearest-even. This is the key step that converts the IEEE 754 bit pattern to a float for the magic constant arithmetic. |
| `SFPMAD` | `a * b + c`, `a * b` | Fused multiply-add (the primary arithmetic workhorse). All float multiplications and additions compile to SFPMAD. Used extensively for the refinement polynomial evaluation and Newton-like correction steps. |
| `SFPSHFT` | `vInt << 8` | Integer left-shift by 8 bits. Restores the full integer result after the scaled magic constant computation (undoes the /256 scaling). |
| `SFPSETSGN` | `sfpi::setsgn(d, a)` | Copy sign bit from input `a` to result `d`. Restores the original sign after computing cbrt of the absolute value. |
| `SFPDIVP2` | `sfpi::addexp(val, 8)` | Add 8 to the exponent field of a float, effectively multiplying by 2^8 = 256. Used in fp32 path to recover `-1/3` from `-1/3/256`. |
| `SFP_STOCH_RND` | `sfpi::float_to_fp16b(y, 0)` | Round FP32 result to FP16B (bfloat16) format. Mode 0 = round-to-nearest-even. Used only in the non-fp32 path before storing. |
| `SFPCONFIG` | `sfpi::vConstFloatPrgmN = val` | Write programmable constant register (in `cube_root_init`). Programs 3 coefficients for the Halley-like refinement polynomial. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile data is read from DEST via `SFPLOAD` and results are written back via `SFPSTORE`. Standard stride-2 addressing: each `dst_reg[0]` accesses 2 physical rows (32 elements). |
| **LREG0-LREG3** (general purpose) | Used implicitly by the compiler for intermediate `vFloat` variables (`a`, `x`, `f`, `y`, `c`, `d`, `t`, `negative_third_256`, `magic`, `negative_third`). The exact register allocation is compiler-determined, but there are up to ~6 live `vFloat` values simultaneously, fitting within the 8 available LREGs. |
| **Programmable Constant 0** (`vConstFloatPrgm0`) | Set to `0x1.c09806p0f` (~1.7523) during init. Degree-0 coefficient of the Halley refinement polynomial. Read as a constant during `calculate_cube_root`. |
| **Programmable Constant 1** (`vConstFloatPrgm1`) | Set to `-0x1.403e6cp0f` (~-1.2510) during init. Degree-1 coefficient of the Halley refinement polynomial. |
| **Programmable Constant 2** (`vConstFloatPrgm2`) | Set to `0x1.04cdb2p-1f` (~0.5094) during init. Degree-2 coefficient of the Halley refinement polynomial. |
| **Fixed Constant `vConstNeg1`** | Value -1.0. Used in fp32 path to compute the residual `c = d*y - 1`. |
| **Fixed Constant `vConst1`** | Value 1.0. Used in fp32 path for the Newton correction `t = 1 + c*(-1/3)`. |

### Address Mode Configuration

The address mode for `SfpuType::cbrt` is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` in the init path. Since `cbrt` does not match any of the special-cased `SfpuType` values (it is not `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, `reciprocal`, etc.), only the default `ADDR_MOD_7` is configured.

**Wormhole B0 and Blackhole** (both identical for cbrt):
```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

This means the hardware does not auto-increment the DEST address between SFPU instructions within a single iteration. The DEST address progression between sfpi rows is handled by the `dst_reg++` abstraction (which maps to the compiler-generated address increment), and the progression between faces is handled by `SETRWC` (Wormhole) or `inc_dst_addr<8>` (Blackhole) in the params dispatch layer.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for CBRT
   **Key Findings**: CBRT uses `eltwise_sfpu.cpp` (default), expands to `cbrt_tile_init()` / `cbrt_tile(idst)`, macro `SFPU_OP_CBRT_INCLUDE`, approx mode false (default)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header exposing `cbrt_tile()` and `cbrt_tile_init()` to the compute kernel
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)`. Uses `APPROX` and `DST_ACCUM_MODE` compile-time constants from the compute kernel.

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>` and `VectorMode::RC`. Init calls `llk_math_eltwise_unary_sfpu_init` with `cube_root_init<APPROXIMATE>`. WH and BH versions are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Implements fast cube root using Moroz et al. magic constant method. Uses SFPI abstractions (vFloat, vInt, dst_reg). Two code paths: fp32 (two refinement steps) and fp16b (one refinement + rounding). WH and BH versions are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that manages face iteration and DEST addressing
   **Key Findings**: For VectorMode::RC, loops over 4 faces calling the SFPU function once per face, with SETRWC (WH) or inc_dst_addr (BH) between faces. WH version uses `set_addr_mod_base()` / `clear_addr_mod_base()` around the SFPU work; BH version does not.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()` only configures `ADDR_MOD_7` with all increments = 0 (cbrt does not match any special-cased SfpuType). Both WH and BH are identical for the default ADDR_MOD_7 path.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI C++ abstractions to underlying SFPU instructions
   **Key Findings**: `abs()` -> `SFPABS`, `int32_to_float()` -> `SFPCAST`, `setsgn()` -> `SFPSETSGN`, `addexp()` -> `SFPDIVP2`, `float_to_fp16b()` -> `SFP_STOCH_RND`, `reinterpret<>()` -> no instruction (type-level cast), `vInt << n` -> `SFPSHFT`.

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand vFloat arithmetic operator mappings and vConst register assignment
   **Key Findings**: `vFloat + vFloat` and `vFloat * vFloat` map to `SFPMAD` (via `flt_add`/`flt_mul` builtins which emit SFPADD/SFPMUL, both MAD-class). `vConst::operator=` maps to `SFPCONFIG` (via `__builtin_rvtt_sfpwriteconfig_v`).

9. **File**: `tt_metal/third_party/tt_ops_code_gen/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware reference for instruction semantics, register layout, and addressing model
   **Key Findings**: Confirmed stride-2 addressing, 8 iterations per face, SFPMAD as the primary arithmetic instruction, SFPCONFIG for programmable constants.
