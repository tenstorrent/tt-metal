## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile_init(); swish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized case: `swish_tile_init()` / `swish_tile(0)` with no explicit template argument; the API header uses `<APPROX>` which resolves to `false` from ComputeConfig |
| Effective SFPU path | `APPROXIMATION_MODE=false` propagated to `calculate_swish<false, 8>()` | The `calculate_swish` function does not branch on `APPROXIMATION_MODE` -- it always uses the same piecewise polynomial/linear/saturation approach regardless of this parameter |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (identical for Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (identical for Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole variant: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `swish_tile_init(); swish_tile(0);`, calling the tile-level API.
2. **API Header** (`swish.h`): `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` within the `MATH()` macro (active only on the TRISC_MATH thread). `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_swish.h`): `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE>(dst_index)` invokes `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`, passing the core SFPU function as a callable. The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST write address, stalls for SFPU availability, then loops over 4 faces (in `VectorMode::RC` mode), calling `calculate_swish<false, 8>()` once per face and advancing the DEST face address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
5. **Core SFPU Implementation** (`ckernel_sfpu_swish.h`): The `calculate_swish<APPROXIMATION_MODE, ITERATIONS>()` function processes 8 sfpi iterations per face, computing `swish(x) = x * sigmoid(x)` via a piecewise approximation of sigmoid.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed. Each face is processed by one invocation of `calculate_swish`, which runs 8 iterations (ITERATIONS=8).
- **Operation invocation**: The parameters dispatch function loops `for (int face = 0; face < 4; face++)`, calling `calculate_swish<false, 8>()` once per iteration. Between faces, the DEST write counter is advanced by 16 physical rows (one face stride).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). Address mode `ADDR_MOD_7` is configured with all increments set to 0 (`srca.incr=0`, `srcb.incr=0`, `dest.incr=0`). This is the same on both Wormhole and Blackhole. The actual DEST address advancement within the kernel is done explicitly by `dst_reg++` in the SFPI code (which translates to incrementing the DEST RWC by `SFP_DESTREG_STRIDE=2` physical rows per iteration).

### Annotated SFPU Kernel Source

The kernel uses **Style A: SFPI-based kernel**. It uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`, `sfpi::vConst1`).

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: take absolute value
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // Horner's method: chain of SFPMAD instructions

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // CC: SFPSETCC(GTE0 on ax-bp1), guarded SFPMAD
        v_endif;

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // CC: SFPSETCC(GTE0 on ax-bp2), guarded SFPLOAD of const 1.0
        v_endif;

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // CC: SFPSETCC(LT0), guarded SFPMAD (1.0 * 1.0 + (-sig_pos))
        v_endif;

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD (x * sig_pos + 0.0) then SFPSTORE back to DEST
        sfpi::dst_reg++; // Advance DEST address by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Description | Usage in swish kernel |
|-------------|-------------|-----------------------|
| `SFPLOAD` | Load data from DEST row pair into LREG | Load `x` from `dst_reg[0]` at each iteration start; also used to load `sfpi::vConst1` (1.0) |
| `SFPSTORE` | Store LREG data back to DEST row pair | Write final `x * sig_pos` result to `dst_reg[0]` |
| `SFPLOADI` | Load 16-bit immediate into LREG | Load float constants: `0.5`, `c1`, `c2`, `c3`, `lin_slope`, `lin_offset`, `bp1`, `bp2`, `0.0` (two SFPLOADI per 32-bit float constant: high 16 bits and low 16 bits) |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC` | Core arithmetic: polynomial evaluation via Horner's method (`c2 + ax * c3`, `c1 + ax * (...)`, `0.5 + ax * (...)`), linear segment (`ax * lin_slope + lin_offset`), negation for `1.0 - sig_pos`, and final `x * sig_pos` product. All float additions and multiplications are compiled to SFPMAD. |
| `SFPABS` | Absolute value | Compute `|x|` from the loaded value |
| `SFPSETCC` | Set condition code based on comparison | Used by `v_if(ax > bp1)`, `v_if(ax > bp2)`, and `v_if(x < 0.0f)` to set per-lane CC bits for conditional execution. The `>` comparison uses subtraction + GTE0 test; the `< 0.0f` test uses LT0 sign bit test. |
| `SFPENCC` | Enable/disable condition code | Emitted at `v_if` entry to enable CC masking and at `v_endif` to disable CC masking (restoring all-lanes-active) |
| `SFPCOMPC` | Complement condition code result | May be emitted as part of `v_if`/`v_endif` CC management |
| `SFPPUSHC` | Push CC state onto stack | May be emitted by `v_if` to save CC state for nested conditional regions |
| `SFPPOPC` | Pop CC state from stack | May be emitted by `v_endif` to restore previous CC state |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST row pair** (via `dst_reg[0]`) | Source and destination for the tile data. Each iteration reads 32 elements (2 physical rows x 16 elements/row) from the current DEST position, processes them, and writes the result back. |
| **LREG0-LREG3** | Used as general-purpose working registers by the SFPI compiler. Hold intermediate values: `x`, `ax` (`|x|`), `sig_pos`, polynomial partial results, comparison temporaries. The compiler allocates these automatically. |
| **LREG4-LREG7** | Available as additional working registers if needed by the compiler for spill/complex expressions. LREG7 may be used for indirect addressing in SFPMAD if the compiler emits such instructions. |
| **Constant registers** | `sfpi::vConst1` maps to Fixed Const 2 (value `1.0`, hex `0x3F800000`). Used for the saturation value and the `1 - sigmoid(|x|)` computation. |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::swish>()` calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`, which configures:

**ADDR_MOD_7** (both Wormhole and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

`SfpuType::swish` does not match any of the special-case `if constexpr` branches in the configure function, so only the default `ADDR_MOD_7` with all-zero increments is configured. This is identical on both Wormhole and Blackhole.

The actual DEST address advancement is handled explicitly within the SFPI kernel code via `dst_reg++` (which increments the DEST RWC by `SFP_DESTREG_STRIDE=2` physical rows per iteration), and between faces by the parameters dispatch function (`TTI_SETRWC` on Wormhole, `math::inc_dst_addr<8>()` on Blackhole).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `eltwise_sfpu.cpp` (default case in `get_compute_kernel_path`), expands to `swish_tile_init(); swish_tile(0);`, and `get_op_approx_mode` returns `false` (default case). Uses the `SFPU_OP_SWISH_INCLUDE` macro guard for conditional inclusion.

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: Trace the API-level tile function and init function signatures
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`. Both are gated by `TRISC_MATH`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: Trace the LLK dispatch layer to the core SFPU function
   **Key Findings**: Dispatches to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`. Default ITERATIONS=8, default vector_mode=VectorMode::RC. Init dispatches to `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Read the core SFPU implementation for full source analysis
   **Key Findings**: Implements swish(x) = x * sigmoid(x) using a piecewise approximation of sigmoid over |x|. Three segments: degree-3 polynomial for |x| <= 2.5, linear interpolation for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0. Uses symmetry: sigmoid(x) = 1 - sigmoid(|x|) for negative x. Pure SFPI style with v_if/v_endif for segment selection.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch (face iteration, DEST advancement, vector mode handling)
   **Key Findings**: Standard VectorMode::RC dispatch: loops over 4 faces, calls the SFPU function once per face, advances DEST by 16 rows between faces using `TTI_SETRWC`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Check address mode configuration for SfpuType::swish
   **Key Findings**: ADDR_MOD_7 configured with all-zero increments (srca=0, srcb=0, dest=0). No special cases for swish.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Confirm how the `APPROX` constant is generated from ComputeConfig
   **Key Findings**: `constexpr bool APPROX = {}` is emitted in `chlkc_descriptors.h` from `desc.get_hlk_math_approx_mode()`, which comes from `math_approx_mode` in the program factory.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Reference for SFPU instruction semantics, addressing model, and register layout
   **Key Findings**: Confirmed stride-2 addressing model, SFPMAD used for all float add/mul, SFPABS for absolute value, CC mechanism for v_if/v_endif.
