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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- switch has only a `default: return false` case |
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` returns `{"swish_tile_init();", "swish_tile({idst});"}` with no template arguments |
| Effective SFPU path | `APPROXIMATION_MODE=false` is propagated through to `calculate_swish<false, 8>()`. However, the kernel does **not** branch on `APPROXIMATION_MODE` -- it ignores this parameter entirely and always uses the same hybrid polynomial+linear sigmoid approximation. | `calculate_swish` in `ckernel_sfpu_swish.h` has `APPROXIMATION_MODE` as a template parameter but never uses it in any `if constexpr` or conditional logic |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`swish_tile(idst)`** (API header `swish.h`): Wraps with `MATH((...))` macro, calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index)`** (LLK dispatch): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_swish<APPROX, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROX>(...)`** (parameters dispatch): Sets DEST write address, stalls for SFPU availability, then iterates over 4 faces calling `calculate_swish<false, 8>()` per face, with `SETRWC` between faces.
4. **`calculate_swish<false, 8>()`** (core SFPU kernel): Executes the SFPI loop processing 8 iterations (one face) of swish computation via piecewise sigmoid approximation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed. The RC branch loops `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face and advancing DEST address between faces.
- **Operation invocation**: `calculate_swish<false, 8>()` is called 4 times total (once per face). Each invocation runs its internal loop of ITERATIONS=8, processing all 8 sfpi rows of one face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice between faces (advancing by 16 physical rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect. The address mode is `ADDR_MOD_7` with `dest.incr=0` (SFPI's `dst_reg++` handles per-iteration advancement internally, not via hardware auto-increment).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`, `sfpi::abs`), so Style A annotation is used. The Wormhole and Blackhole implementations are **identical**.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
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
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST rows into LREG

        // Compute sigmoid(|x|) using degree-3 polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: clear sign bit (FP32 mode)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // Horner's: 3x SFPMAD + SFPLOADI for constants

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // CC: SFPSETCC(GTE) after subtract; guarded SFPMAD
        v_endif;

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // CC: SFPSETCC(GTE) after subtract; guarded SFPMOV from CREG_IDX_1
        v_endif;

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // CC: SFPSETCC(LT0); guarded SFPMAD (1.0 * 1.0 + (-sig_pos))
        v_endif;

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD (x * sig_pos + 0.0), then SFPSTORE back to DEST
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Emitted By | Description |
|-------------|-----------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads 32 elements from current DEST rows into an LREG with format conversion |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Stores LREG contents back to current DEST rows with format conversion |
| `SFPABS` | `sfpi::abs(x)` | Computes absolute value by clearing the sign bit (FP32 mode) |
| `SFPMAD` | `*`, `+`, `-` operators on `vFloat` | Fused multiply-add: `VD = VA * VB + VC`. Used for polynomial evaluation (Horner's method), linear segment computation, negation via sign inversion, and the final `x * sig_pos` multiply |
| `SFPLOADI` | Float literal constants (`0.5f`, `c1`, `c2`, etc.) | Loads 16-bit immediate values into LREGs for use as operands. Two `SFPLOADI` instructions needed per 32-bit float constant (high 16 bits + low 16 bits) |
| `SFPSETCC` | `v_if(ax > bp1)`, `v_if(ax > bp2)`, `v_if(x < 0.0f)` | Sets per-lane condition code based on comparison result. For `>` comparisons: subtract then test GTE0. For `< 0.0f`: test LT0 on sign bit |
| `SFPENCC` | `v_if` / `v_endif` (entry/exit) | Enables CC masking at the start of a `v_if` block and disables it at `v_endif`, restoring unconditional execution |
| `SFPPUSHC` | `v_if` (save CC state) | Pushes current CC state onto the per-lane CC stack for nested conditional support |
| `SFPPOPC` | `v_endif` (restore CC state) | Pops CC state from the per-lane CC stack to restore prior conditional context |
| `SFPCOMPC` | Implicit in `v_if`/`v_endif` pairs when needed | Complements CC.Res for else-branch logic (may be emitted by compiler for CC management) |
| `SFPMOV` | `sig_pos = sfpi::vConst1` (guarded) | Register copy from constant register (CREG_IDX_1 = 1.0) to an LREG, used in the saturation branch |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg[0]`) | Source and destination for tile data. Each iteration reads 32 elements (2 physical DEST rows) and writes 32 elements back. |
| **LREG0-LREG3** (compiler-managed) | Temporary storage for `x`, `ax`, `sig_pos`, and intermediate polynomial/linear results. The SFPI compiler allocates these automatically for `vFloat` variables. |
| **LREG4-LREG7** (compiler-managed) | May be used by the compiler for additional temporaries when register pressure is high (e.g., during Horner's evaluation with 3 nested multiply-adds). |
| **Constant registers** | `CREG_IDX_1` (Fixed Const 2 = 1.0) accessed via `sfpi::vConst1` for saturation and `1 - sigmoid(|x|)` computation. Float literal constants (0.5, 0.2533, -0.01479, -0.00747, 0.0276, 0.855, 2.5, 5.0) are loaded via `SFPLOADI` pairs into LREGs. |
| **CC stack** | Used by 3 independent `v_if`/`v_endif` blocks (not nested). Each block pushes/pops one CC entry. Maximum CC stack depth = 1. |

### Address Mode Configuration

The address mode for the swish SFPU operation is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`:

**ADDR_MOD_7** (applies to both Wormhole and Blackhole -- implementations are identical for this operation):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All auto-increment fields are zero because the SFPI abstraction layer handles DEST address progression internally via `dst_reg++` (which maps to compiler-managed pointer arithmetic on the SFPU address counter), rather than relying on hardware auto-increment. The `SfpuType::swish` does not match any of the special-case `if constexpr` conditions in the address mode configuration, so only the default `ADDR_MOD_7` is set. No `ADDR_MOD_6` is configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `eltwise_sfpu.cpp`, expands to `swish_tile_init()` / `swish_tile(idst)`, approx_mode returns `false` via default case, include guard is `SFPU_OP_SWISH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header defining the tile-level functions `swish_tile()` and `swish_tile_init()`
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, 8>` as the SFPU function and `VectorMode::RC`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation containing the `calculate_swish` function
   **Key Findings**: SFPI-based kernel implementing swish via piecewise sigmoid approximation. Three segments: degree-3 polynomial for |x|<=2.5, linear for 2.5<|x|<=5.0, saturation to 1.0 for |x|>5.0. Symmetry exploited: sigmoid(x) = 1 - sigmoid(|x|) for x<0. APPROXIMATION_MODE template parameter is unused.

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Verify whether Blackhole implementation differs from Wormhole
   **Key Findings**: Implementation is identical to Wormhole (same source code, same algorithm)

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch pattern (face iteration, DEST address progression)
   **Key Findings**: VectorMode::RC loops over 4 faces, calling SFPU function once per face. Between faces, `SETRWC(CR_D, 8)` is called twice (advancing DEST by 16 physical rows). Uses STALLWAIT for SFPU synchronization.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration for SfpuType::swish
   **Key Findings**: swish does not match any special-case conditions; only ADDR_MOD_7 with all increments=0 is configured

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Determine how `APPROX` constant is set during JIT compilation
   **Key Findings**: `constexpr bool APPROX` is emitted into `chlkc_descriptors.h` from `desc.get_hlk_math_approx_mode()`, which is set from `ComputeConfig.math_approx_mode`, which in turn comes from `get_op_approx_mode()` returning false for SWISH

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Reference for SFPU instruction semantics, register model, and addressing
   **Key Findings**: Confirmed stride-2 model, SFPMAD used for all float add/mul, SFPABS for absolute value, CC mechanism for v_if/v_endif blocks

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: Verify that `sfpi::abs()` emits `SFPABS`
    **Key Findings**: `abs(vFloat v)` calls `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)`

11. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Verify `vConst1` maps to `CREG_IDX_1` (Fixed Const 2 = 1.0)
    **Key Findings**: `vConst1(CREG_IDX_1)` where CREG_IDX_1=10 is the fixed constant register holding 1.0
