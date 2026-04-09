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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- `swish_tile_init()` / `swish_tile({idst})` with no explicit template argument; uses the `APPROX` constexpr from `chlkc_descriptors.h` which equals `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `calculate_swish<false, 8>()`. The kernel does not branch on `APPROXIMATION_MODE`; it uses the same piecewise polynomial/linear approximation regardless. | `ckernel_sfpu_swish.h` -- `calculate_swish` has no `if constexpr (APPROXIMATION_MODE)` branch |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (identical on Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (identical on Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole version at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. **`swish_tile(idst)`** (API Header, `swish.h:27`) -- wraps call in `MATH(...)` guard, calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_swish<APPROXIMATE>(dst_index)`** (LLK Dispatch, `llk_math_eltwise_unary_sfpu_swish.h:19`) -- calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::calculate_swish<APPROXIMATE, 8>` as the SFPU functor, `dst_index`, and `VectorMode::RC`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (Parameters Dispatch, `llk_math_eltwise_unary_sfpu_params.h:14`) -- sets DEST write address, stalls until SFPU ready, then loops over 4 faces calling `calculate_swish<false, 8>()` once per face with `SETRWC` advancing between faces.
4. **`calculate_swish<false, 8>()`** (Core SFPU, `ckernel_sfpu_swish.h:35`) -- the inner SFPU kernel that processes one face (8 iterations of 32 elements each = 256 elements).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed. The RC branch loops `for (int face = 0; face < 4; face++)`, calling the SFPU functor once per face.
- **Operation invocation**: The functor `calculate_swish<false, 8>` is called 4 times (once per face). Each call processes 8 SFPU iterations (ITERATIONS=8), covering all 256 elements of a face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` twice between faces (advancing by 8+8=16 physical DEST rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`, `sfpi::vConst1`). Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

namespace ckernel {
namespace sfpu {

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    constexpr float c1 = 0.2533f;    // linear coefficient
    constexpr float c2 = -0.01479f;  // quadratic coefficient
    constexpr float c3 = -0.00747f;  // cubic coefficient

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;  // polynomial-to-linear transition
    constexpr float bp2 = 5.0f;  // linear-to-saturation transition

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair

        // Compute sigmoid(|x|) using degree-3 polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x);  // SFPABS: absolute value
        // Horner form: 0.5 + ax * (c1 + ax * (c2 + ax * c3))
        // Each multiply/add emits SFPMAD instructions
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) {  // CC push, compare ax > 2.5 via SFPSETCC
            sig_pos = ax * lin_slope + lin_offset;  // SFPMAD chain: ax * slope + offset
        }
        v_endif;  // CC pop

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) {  // CC push, compare ax > 5.0
            sig_pos = sfpi::vConst1;  // load fixed constant 1.0 (CREG_IDX_1 = Fixed Const 2)
        }
        v_endif;  // CC pop

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) {  // CC push, compare x < 0 via sign bit test (SFPSETCC)
            sig_pos = sfpi::vConst1 - sig_pos;  // SFPMAD: 1.0 * 1.0 + (-sig_pos), InstrMod[1]=1 for sign inversion
        }
        v_endif;  // CC pop

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos;  // SFPMAD then SFPSTORE: write result back to DEST
        sfpi::dst_reg++;  // advance by 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Description | Usage in swish kernel |
|-------------|-------------|-----------------------|
| `SFPLOAD` | Load data from DEST row pair into LREG | `sfpi::dst_reg[0]` read -- loads 32 elements from current DEST position into an LREG for computation |
| `SFPSTORE` | Store data from LREG back to DEST row pair | `sfpi::dst_reg[0] = ...` write -- stores the final swish result back to DEST |
| `SFPABS` | Compute absolute value (clear sign bit for FP32) | `sfpi::abs(x)` -- computes `|x|` for the symmetric sigmoid approximation |
| `SFPLOADI` | Load 16-bit immediate value into LREG | Loading float literal constants (0.5, 0.2533, -0.01479, -0.00747, 0.0276, 0.855, 2.5, 5.0) into LREGs; each 32-bit float requires two SFPLOADI instructions (upper 16 + lower 16) |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC` | Core arithmetic workhorse: polynomial evaluation via Horner's method (nested multiply-adds), linear interpolation (`ax * slope + offset`), subtraction (`1.0 - sig_pos` via InstrMod sign inversion), final multiplication (`x * sig_pos`) |
| `SFPSETCC` | Set condition code based on comparison | Comparisons in `v_if` blocks: `ax > bp1`, `ax > bp2`, `x < 0.0f`. Sets per-lane CC.Res for predicated execution |
| `SFPENCC` | Enable/disable condition code | Entry/exit of `v_if` blocks: enables CC masking at the start, disables at the end |
| `SFPPUSHC` | Push CC state onto per-lane CC stack | Saves CC state at the start of each `v_if` block for nested conditional support |
| `SFPPOPC` | Pop CC state from per-lane CC stack | Restores CC state at the end of each `v_if`/`v_endif` block |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg[0]`) | Input/output: tile data is read from and written back to the current DEST row pair. Each iteration accesses 2 physical rows (32 elements) at the stride-2 address. |
| **LREG0-LREG3** (implicit, compiler-managed) | Hold intermediate computation values: `x`, `ax`, `sig_pos`, polynomial sub-expressions. The SFPI compiler allocates these transparently. |
| **LREG4-LREG7** (implicit, compiler-managed) | Available for additional temporaries. LREG7 may be used for indirect addressing in certain SFPMAD modes, but this kernel does not use indirect addressing. |
| **Fixed Const 2** (`vConst1`) | Provides the value 1.0f for saturation (`sig_pos = vConst1`) and the negative-x sigmoid correction (`vConst1 - sig_pos`). Accessed via CREG_IDX_1 = constant register index 10. |
| **CC stack** (per-lane, 8 entries) | Used by the three `v_if`/`v_endif` blocks. Each `v_if` pushes one entry; each `v_endif` pops it. Maximum CC stack depth in this kernel is 1 (no nesting). |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::swish>()`, which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`.

**Wormhole B0** (from `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`):

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Notes |
|----------|-----------|-----------|-----------|-------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU addr mod; no auto-increment. DEST addressing is managed by `dst_reg++` in the SFPI kernel and `SETRWC` between faces. |

Swish does not match any of the special-case `if constexpr` branches (topk_local_sort, typecast, unary_max, unary_min, etc.), so only `ADDR_MOD_7` is configured with all-zero increments.

**Blackhole** (from `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`):

Identical configuration -- `ADDR_MOD_7` with `{.srca={.incr=0}, .srcb={.incr=0}, .dest={.incr=0}}`. Swish is not in any special-case branch on Blackhole either.

The DEST address progression is entirely software-managed: within a face, `sfpi::dst_reg++` increments the SFPU address by 1 sfpi row (= 2 physical DEST rows = 32 elements) per loop iteration. Between faces, `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` twice (Blackhole) advances by 16 physical rows to the next face boundary.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for SWISH
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE` macro, `swish_tile_init()` / `swish_tile({idst})` for init/func, default compute kernel `eltwise_sfpu.cpp`, `get_op_approx_mode()` returns `false` for all ops (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header exposing `swish_tile()` and `swish_tile_init()` to compute kernels
   **Key Findings**: Both functions forward to `llk_math_eltwise_unary_sfpu_swish<APPROX>()` and `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()` respectively, guarded by `TRISC_MATH` ifdef

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU function
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_swish<APPROXIMATE, ITERATIONS=8>` as functor. WH and BH versions are identical.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation containing the `calculate_swish` function
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if, abs, vConst1). Implements swish as `x * sigmoid(x)` using a 3-segment piecewise approximation of sigmoid: degree-3 polynomial for |x| <= 2.5, linear interpolation for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0. Symmetric correction for negative x via `1 - sigmoid(|x|)`. WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function managing per-face iteration and DEST progression
   **Key Findings**: `VectorMode::RC` processes all 4 faces with `SETRWC` advancing between faces. SFPU functor called once per face with ITERATIONS=8.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration for SFPU unary operations
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` only configures `ADDR_MOD_7` with all-zero increments. Swish is not in any special-case branch.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` constexpr is generated for compute kernels
   **Key Findings**: `emit_math_scalar_descriptors()` writes `constexpr bool APPROX = {value};` into `chlkc_descriptors.h`, sourced from `hlk_desc.get_hlk_math_approx_mode()` which is set by the program factory's `math_approx_mode` field.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and register layout
   **Key Findings**: Confirmed stride-2 addressing model, SFPMAD as the universal float arithmetic instruction, CC mechanism for predicated execution, SFPABS for absolute value, vConst1 mapping to Fixed Const 2 (1.0).

9. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Verify SFPI abstraction-to-instruction mapping for `abs()`
   **Key Findings**: `sfpi::abs(vFloat)` maps to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)`, confirming `SFPABS` instruction emission.

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Verify `vConst1` definition and `v_if`/`v_endif` macro expansion
    **Key Findings**: `vConst1` is `impl_::vConst<vFloat>(CREG_IDX_1)` where `CREG_IDX_1 = 10` (Fixed Const 2 = 1.0). `v_if` creates a `__vCCCtrl` object managing CC push/pop/if/cond operations.
