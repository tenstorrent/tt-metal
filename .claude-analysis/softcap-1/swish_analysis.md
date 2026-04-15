## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile_init(); swish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized: `swish_tile_init()` / `swish_tile({idst})` with no explicit template args; defaults to `APPROX` which is the JIT-generated constexpr bool from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `calculate_swish<false, 8>()` | The kernel does not branch on `APPROXIMATION_MODE` -- the same piecewise polynomial+linear sigmoid code executes regardless of this flag |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `swish_tile_init(); swish_tile(0);`, calling the API header functions.
2. **API Header** (`swish.h`): `swish_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)))`, gating execution to the MATH thread. `swish_tile_init()` calls `MATH((llk_math_eltwise_unary_sfpu_swish_init<APPROX>()))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_swish.h`): `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE, 8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, 8>, dst_index, VectorMode::RC)`. The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): For `VectorMode::RC`, loops 4 times (one per face), calling `calculate_swish<false, 8>()` each time, with `SETRWC`/`inc_dst_addr` between faces to advance DEST addressing.
5. **Core SFPU Implementation** (`ckernel_sfpu_swish.h`): `calculate_swish<false, 8>()` executes 8 SFPU iterations per face, processing 32 elements (2 DEST rows) per iteration, using piecewise polynomial approximation of sigmoid followed by multiplication with the input.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of a 32x32 tile. The `_llk_math_eltwise_unary_sfpu_params_` function loops `for (int face = 0; face < 4; face++)`.
- **Operation invocation**: `calculate_swish<false, 8>()` is called once per face. Each call runs its internal loop of 8 iterations (ITERATIONS=8), processing one face of 256 elements.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces). On Wormhole, `ADDR_MOD_7` is set with all-zero increments (srca=0, srcb=0, dest=0); face advancement uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice per face (advancing by 16 physical rows = 1 face). On Blackhole, the same `ADDR_MOD_7` with zero increments is configured; face advancement uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`, `sfpi::vConst1`), so Style A (inline-commented source code) is used.

The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

namespace ckernel {
namespace sfpu {

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;    // linear coefficient
    constexpr float c2 = -0.01479f;  // quadratic coefficient
    constexpr float c3 = -0.00747f;  // cubic coefficient

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;   // polynomial-to-linear transition
    constexpr float bp2 = 5.0f;   // linear-to-saturation transition

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST position

        // Compute sigmoid(|x|) using degree-3 polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x);  // SFPABS: clear sign bit, ax = |x|
        // Horner-form evaluation: 0.5 + ax*(c1 + ax*(c2 + ax*c3))
        // Each multiply -> SFPMUL, each add -> SFPMAD (a*1.0+b)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) {  // SFPXFCMPS: compare ax > 2.5, sets CC per-lane
            sig_pos = ax * lin_slope + lin_offset;  // SFPMUL + SFPMAD: linear approximation
        }
        v_endif;  // SFPSETCC/SFPENCC: restore CC state

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) {  // SFPXFCMPS: compare ax > 5.0, sets CC per-lane
            sig_pos = sfpi::vConst1;  // Load constant 1.0 from CREG[10]
        }
        v_endif;  // SFPSETCC/SFPENCC: restore CC state

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) {  // SFPXFCMPS: compare x < 0.0, sets CC per-lane
            sig_pos = sfpi::vConst1 - sig_pos;  // SFPMAD: 1.0 + (-sig_pos) via negation
        }
        v_endif;  // SFPSETCC/SFPENCC: restore CC state

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos;  // SFPMUL + SFPSTORE: multiply and write back to DEST
        sfpi::dst_reg++;  // INCRWC: advance DEST pointer by SFP_DESTREG_STRIDE=2 physical rows
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Builtin / Intrinsic | Description |
|-------------|---------------------|-------------|
| **SFPLOAD** | `__builtin_rvtt_sfpload` | Loads 32 elements (2 physical DEST rows) from the current DEST address into an LREG. Used by `sfpi::dst_reg[0]` reads. |
| **SFPSTORE** | `__builtin_rvtt_sfpstore` | Stores 32 elements from an LREG back to the current DEST address. Used by `sfpi::dst_reg[0] =` writes. |
| **SFPABS** | `__builtin_rvtt_sfpabs` | Computes absolute value by clearing the sign bit of each floating-point element. Used by `sfpi::abs(x)`. |
| **SFPMUL** | `__builtin_rvtt_sfpmul` | Floating-point multiply of two vectors. Used for `ax * c3`, `ax * lin_slope`, `x * sig_pos`, etc. |
| **SFPMAD** | `__builtin_rvtt_sfpadd` | Fused multiply-add (a * 1.0 + b). Used for all `vFloat + vFloat` additions such as `0.5f + ...`, `ax * c2 + ax * c3`, and `vConst1 - sig_pos`. |
| **SFPXFCMPS** | `__builtin_rvtt_sfpxfcmps` | Float compare-scalar: compares each lane of a vFloat against a scalar immediate and sets per-lane condition codes. Used by `ax > bp1`, `ax > bp2`, `x < 0.0f`. |
| **SFPSETCC** | (implicit in `v_if`/`v_endif`) | Saves/restores condition code state for predicated execution blocks. Part of the `v_if`/`v_endif` CC management. |
| **SFPENCC** | (implicit in `v_endif`) | Enables/restores condition codes after a predicated block ends, undoing the CC narrowing from `v_if`. |
| **SFPLOADI** | `__builtin_rvtt_sfpxloadi` | Loads a scalar immediate (e.g., `0.5f`, `0.2533f`, `2.5f`, `5.0f`, `0.0276f`, `0.855f`) into an LREG. Used for all `constexpr float` constants in the kernel. |
| **INCRWC** | `__builtin_rvtt_ttincrwc` | Increments the DEST read/write counter by `SFP_DESTREG_STRIDE` (2 physical rows). Used by `sfpi::dst_reg++`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST (dst_reg[0])** | Input/output: each iteration loads 32 elements from the current DEST position, computes swish, and writes the result back to the same position. DEST addressing advances by `SFP_DESTREG_STRIDE=2` physical rows per iteration via `dst_reg++`. |
| **LREGs (L0-L7)** | Temporary registers used by the compiler for intermediate values: `x` (input copy), `ax` (absolute value), `sig_pos` (sigmoid approximation), polynomial intermediate products, and the final `x * sig_pos` result. The compiler allocates from the pool of 8 LREGs (`SFP_LREG_COUNT=8`) as needed. |
| **CREG[10] (vConst1)** | Constant register holding the value `1.0`. Used for sigmoid saturation (`sig_pos = vConst1`) and for the negative-x correction (`vConst1 - sig_pos`). |

### Address Mode Configuration

For the swish operation, `SfpuType::swish` does not match any special case in `eltwise_unary_sfpu_configure_addrmod()`. Only the default `ADDR_MOD_7` is configured.

**Wormhole B0:**
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```
Source: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`, lines 28-33.

**Blackhole:**
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```
Source: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`, lines 28-33.

Both architectures are identical for swish. The all-zero `ADDR_MOD_7` means DEST auto-increment is disabled at the hardware level; DEST advancement is managed explicitly by `sfpi::dst_reg++` (which emits `INCRWC` instructions) within the kernel loop, and by `SETRWC`/`inc_dst_addr` between faces in the params dispatch layer.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Identify compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE`, maps to `swish_tile_init()` / `swish_tile({idst})`, `get_op_approx_mode()` returns `false` (default), compute kernel is `eltwise_sfpu.cpp` (default)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header layer -- trace how `swish_tile(idst)` calls into the LLK layer
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` gated by `MATH()`. `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer -- trace how the LLK function bridges to the core SFPU implementation
   **Key Findings**: `llk_math_eltwise_unary_sfpu_swish` calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>` as the callable. Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation -- the actual SFPU kernel that computes swish
   **Key Findings**: Piecewise sigmoid approximation using polynomial (degree-3 for |x|<=2.5), linear (2.5<|x|<=5.0), and saturation (|x|>5.0) segments. Uses `sfpi::abs`, `v_if`/`v_endif` for conditional branches, multiplies result by x for final swish. Identical on WH and BH.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand VectorMode::RC loop and DEST address progression
   **Key Findings**: RC mode loops 4 faces, calls SFPU function per face, uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces to advance by 16 physical DEST rows (1 face).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function -- understand ADDR_MOD configuration for swish
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` only sets `ADDR_MOD_7` with all-zero increments. No special ADDR_MOD_6 for swish.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Blackhole init function -- verify ADDR_MOD matches Wormhole
   **Key Findings**: Identical ADDR_MOD_7 configuration. Blackhole adds `SfpuType::reciprocal` to the ADDR_MOD_6 special-case list, but swish is not affected.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Blackhole params dispatch -- compare with Wormhole params dispatch
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_start_`/`_done_()` wrappers and `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of direct `TTI_SETRWC` calls. Functionally equivalent to Wormhole.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI abstraction library -- understand how vFloat operations map to SFPU instructions
   **Key Findings**: `vFloat + vFloat` -> `__builtin_rvtt_sfpadd` (SFPMAD), `vFloat * vFloat` -> `__builtin_rvtt_sfpmul` (SFPMUL), `vFloat > float` -> `__builtin_rvtt_sfpxfcmps` (SFPXFCMPS), `dst_reg++` -> `__builtin_rvtt_ttincrwc` (INCRWC).

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: SFPI library functions -- understand `sfpi::abs` implementation
    **Key Findings**: `sfpi::abs(vFloat)` -> `__builtin_rvtt_sfpabs(v, SFPABS_MOD1_FLOAT)` (SFPABS instruction).

11. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Understand how `APPROX` constexpr bool is generated from `math_approx_mode`
    **Key Findings**: Line 394: `constexpr bool APPROX = {};` is generated from `desc.get_hlk_math_approx_mode()`, which comes from the `math_approx_mode` field set in `UnaryProgramFactory`.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU hardware model constants and addressing
    **Key Findings**: Confirmed tile geometry (32x32, 4 faces of 16x16), DEST stride-2 model (SFP_DESTREG_STRIDE=2), 8 iterations per face, 32 elements per iteration, SFP_LREG_COUNT=8.
