## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default case in `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized: `swish_tile_init()` / `swish_tile({idst})` with default template args; API header uses `<APPROX>` which resolves to the JIT-generated `constexpr bool APPROX = false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` but the parameter is never referenced in the function body; the same hybrid polynomial+piecewise-linear approximation executes unconditionally | `calculate_swish<APPROXIMATION_MODE, ITERATIONS>` in `ckernel_sfpu_swish.h` -- no `if constexpr (APPROXIMATION_MODE)` branch exists |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (identical across architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (identical across architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `swish_tile(0)`.
2. **API Header** (`swish.h`): `swish_tile(uint32_t idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` inside the `MATH(...)` gate (only compiled for TRISC_MATH).
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_swish.h`): `llk_math_eltwise_unary_sfpu_swish<APPROXIMATE, ITERATIONS=8>(dst_index, vector_mode=VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls for SFPU readiness, then loops 4 times (for `VectorMode::RC`) calling `calculate_swish<false, 8>()` once per face, with `TTI_SETRWC` advancing between faces.
5. **Core SFPU** (`ckernel_sfpu_swish.h`): `calculate_swish<false, 8>()` executes the 8-iteration SFPI loop processing one face (256 elements).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls `calculate_swish()` once per face (4 times total). Each call executes `ITERATIONS=8` iterations of the inner loop.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). Address mode `ADDR_MOD_7` is configured with all-zero increments (`srca=0, srcb=0, dest=0`) on both Wormhole and Blackhole -- the SFPI `dst_reg++` abstraction handles DEST pointer advancement internally.

### Annotated SFPU Kernel Source
The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`, `sfpi::abs`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;       // degree-1 coefficient
    constexpr float c2 = -0.01479f;     // degree-2 coefficient
    constexpr float c3 = -0.00747f;     // degree-3 coefficient

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;         // polynomial-to-linear transition
    constexpr float bp2 = 5.0f;         // linear-to-saturation transition

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {  // 8 iterations per face, 32 elements each
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST position

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x);     // SFPABS: absolute value (clear sign bit)
        // Horner's method: 0.5 + ax*(c1 + ax*(c2 + ax*c3))
        // Each + and * on vFloat emits SFPMAD (fused multiply-add)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) {                    // SFPMAD(ax - bp1) + SFPSETCC(GTE0) + SFPPUSHC/SFPENCC
            sig_pos = ax * lin_slope + lin_offset;  // SFPMAD: linear interpolation
        }
        v_endif;                             // SFPPOPC: restore CC state

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) {                    // SFPMAD(ax - bp2) + SFPSETCC(GTE0) + SFPPUSHC/SFPENCC
            sig_pos = sfpi::vConst1;         // Load constant 1.0 from fixed constant register
        }
        v_endif;                             // SFPPOPC: restore CC state

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) {                    // SFPSETCC(LT0 on x) + SFPPUSHC/SFPENCC
            sig_pos = sfpi::vConst1 - sig_pos;  // SFPMAD: 1.0 + (-sig_pos), sign inversion via SFPMOV
        }
        v_endif;                             // SFPPOPC: restore CC state

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos;     // SFPMAD(x * sig_pos + 0.0) then SFPSTORE to DEST
        sfpi::dst_reg++;                     // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

### SFPU Instructions Used
| Instruction | Emitted By | Description |
|-------------|-----------|-------------|
| `SFPLOAD` | `dst_reg[0]` read | Loads 32 elements from current DEST position into an LREG |
| `SFPLOADI` | `vFloat(float)` constructors for constants (0.5f, c1, c2, c3, bp1, bp2, lin_slope, lin_offset, 0.0f) | Loads 16-bit immediate to LREG (two instructions for 32-bit float) |
| `SFPABS` | `sfpi::abs(x)` | Computes absolute value by clearing the sign bit (FP32 mode) |
| `SFPMAD` | `vFloat + vFloat`, `vFloat * vFloat`, `vFloat - vFloat` | Fused multiply-add: `VD = VA * VB + VC`. Used for all arithmetic: polynomial evaluation (Horner chain), linear interpolation, subtraction (via sign inversion on addend), and the final `x * sig_pos` |
| `SFPMOV` | Negation (`-sig_pos` in `vConst1 - sig_pos`) | Register copy with optional sign complement (`SFPMOV_MOD1_COMPSIGN`) |
| `SFPSETCC` | Float comparisons (`ax > bp1`, `ax > bp2`, `x < 0.0f`) | Sets per-lane CC.Res based on comparison result (sign bit test after subtraction) |
| `SFPPUSHC` | `v_if` macro | Pushes current CC state onto the CC stack to save context before conditional block |
| `SFPPOPC` | `v_endif` macro | Pops CC state from stack, restoring previous conditional context |
| `SFPENCC` | `v_if` / `v_endif` CC management | Enables/disables CC masking for conditional execution blocks |
| `SFPSTORE` | `dst_reg[0] = ...` write | Stores LREG value back to DEST at current position |

### SFPU Register Usage
| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General-purpose temporaries used by the SFPI compiler for intermediate values: `x`, `ax`, `sig_pos`, polynomial sub-expressions, comparison results. The SFPI compiler allocates these automatically. |
| **DEST (dst_reg)** | Source and destination for tile data. Each iteration reads 32 elements from DEST, computes swish, and writes 32 elements back. The `dst_reg++` abstraction advances by `SFP_DESTREG_STRIDE=2` physical DEST rows per iteration. |
| **Fixed Constant Register (CREG_IDX_1)** | Provides the value 1.0 (accessed via `sfpi::vConst1`). Used for the saturation case (`sig_pos = 1.0`) and for the negative-x correction (`1.0 - sig_pos`). |
| **CC Stack** | Three `v_if`/`v_endif` blocks each push/pop one CC entry. Maximum CC stack depth during execution is 1 (the blocks are sequential, not nested). |

### Address Mode Configuration
The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` within `llk_math_eltwise_unary_sfpu.h`.

**Both Wormhole and Blackhole** use identical configuration:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode; DEST advancement is handled by the SFPI `dst_reg++` abstraction (which internally manages the RWC), not by hardware auto-increment. |

The `swish` SfpuType does not match any special-case `if constexpr` branch in the address mode configuration, so only the default `ADDR_MOD_7` (all-zero increments) is set. Between faces, the params dispatch uses `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` twice (advancing by 16 physical rows = 1 face) to position the DEST pointer for the next face.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine dispatch path -- compute kernel, init/func strings, approx mode
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE`, `swish_tile_init()`/`swish_tile({idst})`, compute kernel `eltwise_sfpu.cpp`, approx mode `false`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header -- trace from tile-level call to LLK
   **Key Findings**: `swish_tile()` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch -- bridge from API to ckernel
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)` with `VectorMode::RC` default

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation -- the actual compute kernel
   **Key Findings**: Hybrid 3-segment sigmoid approximation: degree-3 polynomial for |x|<=2.5, linear for 2.5<|x|<=5.0, saturation to 1.0 for |x|>5.0. Uses `sfpi::abs`, `v_if`/`v_endif`, Horner polynomial evaluation. APPROXIMATION_MODE template parameter exists but is never used in the function body.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand face iteration and DEST progression
   **Key Findings**: VectorMode::RC loops 4 faces, calls sfpu_func once per face, uses TTI_SETRWC to advance between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `ADDR_MOD_7` with all-zero increments for swish (no special case), identical on WH and BH

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Verify `sfpi::abs` maps to `SFPABS`
   **Key Findings**: `abs(vFloat v)` calls `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)`

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI abstractions, operator overloads, v_if/v_endif macros, vConst1
   **Key Findings**: `v_if` pushes CC stack, evaluates condition, guards block; `vFloat + vFloat` -> `flt_add` -> SFPMAD; `vConst1` is fixed constant register index CREG_IDX_1 (value 1.0)

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model reference
   **Key Findings**: Stride-2 addressing, 8 iterations per face, SFPMAD semantics, CC mechanism

10. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Verify how `APPROX` constant is generated
    **Key Findings**: Line 394: `constexpr bool APPROX = {math_approx_mode}` is JIT-generated from `hlk_desc.get_hlk_math_approx_mode()`
