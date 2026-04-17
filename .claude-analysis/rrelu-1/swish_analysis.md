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
| Template parameter (SFPU_OP_CHAIN) | none (default template args used) | `get_op_init_and_func_default()` returns `{"swish_tile_init();", "swish_tile(0);"}` -- no parameterized version exists for SWISH |
| Effective SFPU path | `APPROXIMATION_MODE=false` for `calculate_swish<false, 8>()`. However, the kernel does not branch on `APPROXIMATION_MODE` -- the same piecewise sigmoid code executes regardless. | `calculate_swish` in `ckernel_sfpu_swish.h` has no `if constexpr(APPROXIMATION_MODE)` branch |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (identical for Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (identical for Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole variant at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. **`swish_tile(idst)`** (API header, `swish.h:27`) -- wraps the call in the `MATH((...))` macro (executes only on the math RISC-V thread) and calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_swish<APPROXIMATE, 8>(dst_index, VectorMode::RC)`** (LLK dispatch, `llk_math_eltwise_unary_sfpu_swish.h:19`) -- invokes `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with the core function `ckernel::sfpu::calculate_swish<APPROXIMATE, 8>` as the callable, plus `dst_index` and `VectorMode::RC`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (params dispatch, `llk_math_eltwise_unary_sfpu_params.h:14`) -- sets DEST write address for the tile, configures address mode base, stalls until SFPU is ready, then loops over 4 faces (for `VectorMode::RC`), calling `calculate_swish<false, 8>()` once per face and advancing the DEST face address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
4. **`calculate_swish<false, 8>()`** (core SFPU, `ckernel_sfpu_swish.h:35`) -- the innermost function that performs the actual SFPU computation, iterating 8 times per face (one iteration per sfpi row = 32 elements).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed, covering the full 32x32 tile (1024 elements).
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_swish<false, 8>()` once per face. Each call processes 8 sfpi iterations (ITERATIONS=8), covering one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). On Wormhole, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is issued twice between faces (each advancing DEST by 8 sfpi rows = 16 physical rows), effectively jumping over the gap from the kernel's 8-row iteration range to the next face boundary. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`). Style A (inline-commented source code) is used.

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
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face, 32 elements per iteration
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: compute absolute value (clear sign bit)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // Horner's method: chain of SFPMAD (multiply-add). Evaluates 0.5 + ax*(0.2533 + ax*(-0.01479 + ax*(-0.00747)))

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // CC: SFPSETCC tests ax > 2.5, guarded SFPMAD computes linear approx
        v_endif; // CC: SFPPOPC restores CC state

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // CC: SFPSETCC tests ax > 5.0, guarded SFPLOADI/SFPMOV loads 1.0 from constant register
        v_endif; // CC: SFPPOPC restores CC state

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // CC: SFPSETCC tests x < 0 (sign bit), guarded SFPMAD computes 1.0 - sig_pos
        v_endif; // CC: SFPPOPC restores CC state

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD: multiply x * sig_pos, then SFPSTORE: write result back to DEST
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | SFPI Source Construct | Description |
|-------------|----------------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG for SFPU computation |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Stores 32 elements from an LREG back to the current DEST row pair |
| `SFPABS` | `sfpi::abs(x)` | Computes element-wise absolute value by clearing the sign bit (float mode) |
| `SFPMAD` | `vFloat * vFloat`, `vFloat + vFloat`, `a * b + c` expressions | Fused multiply-add: `VD = VA * VB + VC`. Used for all polynomial evaluation (Horner's method), linear interpolation, subtraction (`1.0 - sig_pos`), and final multiplication (`x * sig_pos`). There is no dedicated float add instruction; additions compile as `SFPMAD(a, 1.0, b)`. |
| `SFPLOADI` | `constexpr float` literal loads (0.5f, c1, c2, c3, lin_slope, lin_offset, bp1, bp2, 0.0f) | Loads a 16-bit immediate value into an LREG. Used to materialize floating-point constants. Two `SFPLOADI` instructions needed for a full 32-bit float. |
| `SFPSETCC` | `v_if(ax > bp1)`, `v_if(ax > bp2)`, `v_if(x < 0.0f)` (comparison part) | Sets the per-lane condition code (CC.Res) based on a comparison. For `>` comparisons, the SFPI compiler emits a subtract followed by sign-bit test; for `< 0.0f`, it tests the sign bit directly. |
| `SFPENCC` | `v_if` / `v_endif` (CC enable/disable) | Enables or disables the CC masking system. At `v_if` entry, enables CC; at final `v_endif` exit, disables CC so all lanes become active again. |
| `SFPPUSHC` | `v_if` (CC stack push) | Pushes the current CC state onto the per-lane CC stack to enable nested conditional scoping. Each `v_if` pushes, each `v_endif` pops. |
| `SFPPOPC` | `v_endif` (CC stack pop) | Pops the CC state from the per-lane CC stack, restoring the prior conditional scope. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (current sfpi row pair) | Source and destination for tile data. `dst_reg[0]` reads/writes the current 32-element pair. `dst_reg++` advances to the next pair. |
| **LREG0-LREG3** (general purpose) | Used by the compiler to hold intermediate values: `x`, `ax`, `sig_pos`, and temporary results from polynomial evaluation and comparisons. The SFPI compiler allocates LREGs automatically for `vFloat` variables. |
| **LREG4-LREG7** (general purpose) | Available for compiler use if needed. LREG7 can serve as an indirect address register for SFPMAD, but this kernel does not use indirect addressing. |
| **Constant Register: Fixed Const 1** (`vConst0`, index 9) | 0.0f -- used implicitly in `SFPMAD` for addition operations (multiply by 1.0, add term) and in the `< 0.0f` comparison. |
| **Constant Register: Fixed Const 2** (`vConst1`, index 10) | 1.0f -- used explicitly for saturation (`sig_pos = vConst1`) and for the symmetry correction (`vConst1 - sig_pos`). |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` during `llk_math_eltwise_unary_sfpu_init`. Since `SfpuType::swish` does not match any of the special-cased `SfpuType` values in the `if constexpr` branches (which cover `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical for this operation):**

| Address Mode | Field | Value | Purpose |
|-------------|-------|-------|---------|
| `ADDR_MOD_7` | `srca.incr` | 0 | No auto-increment on source A |
| `ADDR_MOD_7` | `srcb.incr` | 0 | No auto-increment on source B |
| `ADDR_MOD_7` | `dest.incr` | 0 | No auto-increment on DEST |

The DEST address advancement within the face is handled by the SFPI `dst_reg++` abstraction (which emits implicit pointer increment instructions), not by the `ADDR_MOD` auto-increment. Between faces, the params dispatch uses `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole) to jump to the next face's starting row.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine the compute kernel path, SFPU_OP_CHAIN macro expansion, and approximation mode for the SWISH operation
   **Key Findings**: SWISH uses `SFPU_OP_SWISH_INCLUDE` guard, `swish_tile_init()` / `swish_tile(idst)` dispatch, `eltwise_sfpu.cpp` compute kernel, `math_approx_mode=false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header that exposes `swish_tile()` and `swish_tile_init()` to the compute kernel
   **Key Findings**: Passes `APPROX` template parameter to `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)` and `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_swish<APPROXIMATE, ITERATIONS>`, uses `VectorMode::RC` default, `ITERATIONS=8` default

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU kernel implementation containing the actual computation
   **Key Findings**: Implements piecewise sigmoid approximation with 3 segments (degree-3 polynomial for |x|<=2.5, linear for 2.5<|x|<=5.0, saturation at 1.0 for |x|>5.0), then computes swish = x * sigmoid(x). Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that manages face iteration and DEST addressing for Wormhole
   **Key Findings**: For VectorMode::RC, iterates 4 faces, calls sfpu_func once per face, advances DEST address by 2x SETRWC(8) between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function for Blackhole
   **Key Findings**: Same face iteration pattern as Wormhole but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of `TTI_SETRWC` for face advancement

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration for Wormhole
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` configures only `ADDR_MOD_7` with all-zero increments (swish does not match any special-cased SfpuType)

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how the `APPROX` constexpr bool is generated at JIT build time
   **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = {math_approx_mode};` from `emit_math_scalar_descriptors()`, sourced from `desc.get_hlk_math_approx_mode()`

9. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Understand what SFPU instruction `sfpi::abs()` maps to
   **Key Findings**: `sfpi::abs(vFloat)` maps to `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)` which emits the `SFPABS` instruction

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware model reference for instruction semantics, register layout, and addressing
    **Key Findings**: Confirmed stride-2 model (32 elements per sfpi row), SFPMAD for all float add/mul, SFPABS for absolute value, CC mechanism for v_if/v_endif, constant register mappings (vConst1 = Fixed Const 2 = 1.0f)

11. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Understand v_if/v_endif macro expansion and vConst1 constant register mapping
    **Key Findings**: `v_if(x)` expands to CC push + enable + condition test, `v_endif` pops CC stack. `vConst1` maps to `CREG_IDX_1` = constant register index 10 = Fixed Const 2 = 1.0f
