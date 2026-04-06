## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSIGMOID`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardsigmoid_tile_init(); hardsigmoid_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSIGMOID)` in `unary_op_utils.cpp` -- returns `false` via `default` case (switch has only `default: return false`) |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterization) | `get_op_init_and_func_default()` -- `hardsigmoid_tile_init()` / `hardsigmoid_tile(0)` with no template arguments; the API header defaults to `APPROX` which is the JIT-generated `constexpr bool APPROX = false` |
| Effective SFPU path | `APPROXIMATION_MODE = false` in `calculate_hardsigmoid<false, 8>()` | The kernel has no `if constexpr (APPROXIMATION_MODE)` branches -- the template parameter is accepted but not used to select different code paths. The same linear computation runs regardless of the approximation mode. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardsigmoid_tile_init(); hardsigmoid_tile(0);`, invoking the API functions.
2. **API Header** (`hardsigmoid.h`): `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` inside the `MATH()` macro (active only on the math RISC-V thread).
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardsigmoid.h`): `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROXIMATE>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up the DEST write address, stalls until SFPU is ready, then loops over 4 faces (VectorMode::RC), calling `calculate_hardsigmoid<false, 8>()` once per face, advancing the DEST address by 16 physical rows between faces via `TTI_SETRWC` (WH) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (BH).
5. **Core SFPU** (`ckernel_sfpu_hardsigmoid.h`): `calculate_hardsigmoid<false, 8>()` runs the 8-iteration loop processing one face (256 elements), computing `max(0, min(1, x/6 + 0.5))` per element.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed, covering the full 32x32 = 1024 elements.
- **Operation invocation**: The dispatch loops over 4 faces. For each face, `calculate_hardsigmoid<false, 8>()` is called once, which internally iterates 8 times (ITERATIONS=8), processing 32 elements per iteration (2 physical DEST rows x 16 elements/row), totaling 256 elements per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces). The address mode is ADDR_MOD_7 on both Wormhole and Blackhole, configured with `.dest = {.incr = 0}` (no auto-increment from the address mode -- the SFPI `dst_reg++` handles iteration advancement via compiler-emitted DEST offset updates).

### Annotated SFPU Kernel Source
The kernel uses **Style A: SFPI-based kernel** with `vFloat`, `dst_reg`, `v_if`/`v_endif` abstractions.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h

namespace ckernel::sfpu {

// hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
// Piecewise linear:
//   x <= -3  =>  0
//   x >= 3   =>  1
//   else     =>  x * (1/6) + 0.5
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f; // ~0.16667f, loaded as immediate into LREG

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];         // SFPLOAD: load 32 elements from current DEST rows into LREG
        sfpi::vFloat result = x * one_sixth + 0.5f; // SFPMAD: result = x * (1/6) + 0.5 (fused multiply-add)

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }     // CC block: SFPSETCC(LT0) + SFPLOADI(0.0) guarded by CC
        v_endif;                                     // CC restore via SFPPOPC
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // CC block: compare > 1.0, SFPMOV from const reg
        v_endif;                                     // CC restore via SFPPOPC

        sfpi::dst_reg[0] = result;                   // SFPSTORE: write 32 elements back to DEST
        sfpi::dst_reg++;                             // advance DEST pointer by 1 sfpi row (2 physical rows)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| `SFPLOAD` | Load 32 elements from current DEST rows into an LREG. Emitted by `sfpi::vFloat x = sfpi::dst_reg[0]`. |
| `SFPLOADI` | Load 16-bit immediate constants into LREGs. Used for the `one_sixth` (1/6) and `0.5f` constants, and the `0.0f` clamping value. Two SFPLOADI instructions are needed per 32-bit float constant (high 16 bits + low 16 bits). |
| `SFPMAD` | Fused multiply-add: `result = x * one_sixth + 0.5f`. This is the core linear transformation. The SFPI expression `x * one_sixth + 0.5f` compiles to a single SFPMAD instruction computing `VA * VB + VC`. |
| `SFPSETCC` | Set condition code per-lane. Used twice: once for `result < 0.0f` (mode `LREG_LT0`, checking sign bit) and once for the `result > vConst1` comparison (implemented via subtraction + sign test). |
| `SFPPUSHC` | Push current CC state onto the CC stack. Emitted by `v_if` to save CC state before entering a conditional block. |
| `SFPPOPC` | Pop CC state from the CC stack. Emitted by `v_endif` to restore CC state after a conditional block. |
| `SFPCOMPC` | Complement CC.Res for else-branch logic. May be emitted internally by the `v_if` / comparison infrastructure to implement the `>` comparison (which requires negating a `<=` test). |
| `SFPMOV` | Register-to-register copy. Used when assigning `result = sfpi::vConst1` inside the second `v_if` block -- copies the constant register value (1.0f) into the result LREG. |
| `SFPSTORE` | Store 32 elements from an LREG back to DEST rows. Emitted by `sfpi::dst_reg[0] = result`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows (current sfpi address)** | Source of input `x` via `SFPLOAD`, and destination for `result` via `SFPSTORE`. The `dst_reg++` advances through 8 sfpi addresses per face, covering 256 elements. |
| **LREG0-LREG3** (general purpose) | Used as temporaries by the compiler for holding `x`, `result`, intermediate computation results, and immediate constants (`one_sixth`, `0.5f`, `0.0f`). The exact LREG assignment is compiler-determined. |
| **Fixed Const 2** (value: 1.0f) | Accessed via `sfpi::vConst1`. Used in the upper clamping comparison (`result > 1.0f`) and as the clamping value (`result = 1.0f`). |
| **CC register (per-lane)** | Used for predicated execution in the two `v_if` clamping blocks. CC.En and CC.Res control which lanes execute the clamping assignments. |
| **CC stack** | Used by `v_if`/`v_endif` (SFPPUSHC/SFPPOPC) to save and restore CC state across the two clamping blocks. |

### Address Mode Configuration

The hardsigmoid operation uses ADDR_MOD_7 on both Wormhole and Blackhole, configured identically:

```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This is the default SFPU address mode for unary operations. The `.dest = {.incr = 0}` means the hardware address mode does not auto-increment the DEST register pointer between SFPU instructions. Instead, DEST addressing within each face is managed by the SFPI compiler via explicit `dst_reg++` operations (which compile to offset adjustments in SFPLOAD/SFPSTORE address fields). Between faces, the params dispatch function advances the DEST pointer by 16 physical rows (one face height) using:
- **Wormhole**: `TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)` called twice (2 x 8 = 16 physical rows)
- **Blackhole**: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice

The `SfpuType::hardsigmoid` does not match any special-case `if constexpr` branch in `eltwise_unary_sfpu_configure_addrmod()`, so no additional address modes (e.g., ADDR_MOD_6) are configured.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDSIGMOID
   **Key Findings**: Compute kernel is `eltwise_sfpu.cpp` (default), init/func are `hardsigmoid_tile_init()` / `hardsigmoid_tile(0)`, `get_op_approx_mode()` returns `false` for all ops via default case

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: Trace API-level tile function to LLK dispatch
   **Key Findings**: `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` via the MATH macro

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: Trace LLK dispatch to core SFPU function
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardsigmoid<APPROXIMATE, 8>, dst_index, VectorMode::RC)`. WH and BH files are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Read core SFPU kernel implementation
   **Key Findings**: SFPI-based kernel computing `max(0, min(1, x/6 + 0.5))` with two v_if clamp blocks. WH and BH implementations are identical. Does not branch on APPROXIMATION_MODE.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand parameters dispatch and face iteration pattern
   **Key Findings**: VectorMode::RC loops over 4 faces, calling the SFPU function once per face, with TTI_SETRWC advancing DEST by 16 physical rows between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration and init function
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()` sets ADDR_MOD_7 with `.dest = {.incr = 0}`. No special-case branches for hardsigmoid.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Verify BH ADDR_MOD configuration matches WH
   **Key Findings**: Identical ADDR_MOD_7 configuration. BH params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of raw `TTI_SETRWC`.

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` compile-time constant is set from `math_approx_mode`
   **Key Findings**: `constexpr bool APPROX = {}` is generated from `desc.get_hlk_math_approx_mode()`, which comes from `math_approx_mode` in ComputeConfig

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model for instruction semantics, register layout, and addressing
   **Key Findings**: SFPMAD is the core FMA instruction (no dedicated add), vConst1 maps to Fixed Const 2 = 1.0f, stride-2 addressing model confirmed

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Verify vConst1 mapping
    **Key Findings**: `vConst1` is `vConst<vFloat>(CREG_IDX_1)` where CREG_IDX_1 = 10, mapping to Fixed Const 2 = 1.0f
