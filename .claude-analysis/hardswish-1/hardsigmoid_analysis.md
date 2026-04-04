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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSIGMOID)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` returns `{"hardsigmoid_tile_init();", "hardsigmoid_tile({idst});"}` -- no template parameter in the init/func strings |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_hardsigmoid` | The API header uses `APPROX` (JIT-generated `constexpr bool` from `math_approx_mode`), which becomes `false`. The kernel does not use `APPROXIMATION_MODE` in any `if constexpr` branch, so the code path is identical regardless. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`hardsigmoid_tile(idst)`** (API header `hardsigmoid.h:27`): Wraps the call in `MATH(...)` and invokes `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`.

2. **`llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX, 8>(dst_index)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_hardsigmoid.h:18-22`): Delegates to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_hardsigmoid<APPROX, 8>, dst_index, VectorMode::RC)`.

3. **`_llk_math_eltwise_unary_sfpu_params_<false>(...)`** (params dispatch): Sets the DEST write address, sets address mode base (Wormhole) or starts SFPU (Blackhole), stalls SFPU until MATH is ready, then loops over 4 faces in `VectorMode::RC`, calling `calculate_hardsigmoid<false, 8>()` once per face and advancing the DEST face address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).

4. **`calculate_hardsigmoid<false, 8>()`** (core SFPU `ckernel_sfpu_hardsigmoid.h:18`): Executes 8 iterations per face, each processing 32 elements via SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile (faces 0, 1, 2, 3), covering the full 32x32 tile.
- **Operation invocation**: In `VectorMode::RC`, the params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_hardsigmoid<false, 8>()` once per face. Each invocation runs 8 SFPU iterations (the `ITERATIONS` template parameter), processing one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (the SFPU kernel manages its own address progression via `dst_reg++`); between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance by 16 physical DEST rows (one face stride). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect. Within a face, `dst_reg++` advances 1 sfpi row = 2 physical DEST rows = 32 elements per iteration.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used. The implementation is identical on Wormhole B0 and Blackhole.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h

namespace ckernel::sfpu {

// hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f; // ~0.16667f, loaded via SFPLOADI into an LREG

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];             // SFPLOAD: load 32 elements from current DEST row into LREG
        sfpi::vFloat result = x * one_sixth + 0.5f;    // SFPMAD: result = x * (1/6) + 0.5 (fused multiply-add)

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }         // SFPSETCC (LT0 test on result), CC-guarded SFPLOADI/SFPMOV to set 0.0
        v_endif;                                         // SFPENCC to restore all-lanes-active
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // SFPSETCC (compare with 1.0), CC-guarded assignment of 1.0 (Fixed Const 2)
        v_endif;                                         // SFPENCC to restore all-lanes-active

        sfpi::dst_reg[0] = result;                      // SFPSTORE: write result back to current DEST row
        sfpi::dst_reg++;                                 // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| `SFPLOAD` | Load 32 elements from current DEST rows into an LREG. Emitted by `sfpi::vFloat x = sfpi::dst_reg[0]`. |
| `SFPLOADI` | Load 16-bit immediate constants into LREGs. Used to materialize `one_sixth` (~0.16667f), `0.5f`, and `0.0f` as FP32 values (requires two SFPLOADI for full 32-bit). |
| `SFPMAD` | Fused multiply-add: computes `x * one_sixth + 0.5f` in a single instruction (`VD = VA * VB + VC`). Also used implicitly for float additions (add = MAD with multiplier 1.0). |
| `SFPSETCC` | Set per-lane condition code based on comparison. Emitted by `v_if(result < 0.0f)` (tests sign bit, mode `LREG_LT0`) and `v_if(result > vConst1)` (comparison against 1.0, involves subtraction + sign test). |
| `SFPENCC` | Enable/disable condition code masking. Emitted at the start of each `v_if` block (to enable CC) and by each `v_endif` (to disable CC and restore all-lanes-active). |
| `SFPCOMPC` | Complement condition code for else-branch logic. May be emitted internally by `v_if` CC management depending on the comparison polarity. |
| `SFPPUSHC` | Push CC state onto the per-lane CC stack. Emitted by `v_if` to save the current CC state before entering a conditional block. |
| `SFPPOPC` | Pop CC state from the per-lane CC stack. Emitted by `v_endif` to restore the previous CC state after exiting a conditional block. |
| `SFPMOV` | Register-to-register copy within LREGs. Used to move constant values (0.0, 1.0) into the result register under CC masking. |
| `SFPSTORE` | Store LREG contents back to the current DEST rows. Emitted by `sfpi::dst_reg[0] = result`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** (LREGS1 bank) | Used as temporaries by the SFPI compiler for: the loaded input value `x`, the intermediate/final `result`, and materialized constants (`one_sixth`, `0.5f`, `0.0f`). The SFPI compiler allocates LREGs automatically; the exact mapping depends on compiler register allocation but fits within 4 registers. |
| **DEST rows** | Source and destination for tile data. Each `dst_reg[0]` access reads/writes 32 elements (2 physical DEST rows x 16 elements/row). The DEST row pointer auto-increments via `dst_reg++`. |
| **Fixed Const 2** (`vConst1`) | Hardware constant register with value 1.0 (`0x3F800000`). Used directly in the comparison `result > sfpi::vConst1` and the clamping assignment `result = sfpi::vConst1`, avoiding the need to load 1.0 via SFPLOADI. |
| **CC stack** | 2 entries used (one per `v_if`/`v_endif` pair). Each `v_if` pushes one entry; each `v_endif` pops it. Since the two `v_if` blocks are sequential (not nested), the CC stack depth never exceeds 1 at any point. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()`, called from the init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardsigmoid>()`.

**Both Wormhole B0 and Blackhole** configure the same address mode:

```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

The `hardsigmoid` SfpuType does not match any of the special-case `if constexpr` branches (which configure `ADDR_MOD_6` for `topk_local_sort`, `typecast`, `reciprocal`, or `unary_max/min` variants), so only the default `ADDR_MOD_7` with all-zero increments is set.

This means the hardware does NOT auto-increment DEST addressing between SFPU iterations. Instead, DEST address progression is handled entirely by the SFPI abstraction layer: `dst_reg++` in the kernel loop explicitly advances the SFPU's internal DEST row pointer by `SFP_DESTREG_STRIDE = 2` physical rows per iteration. Between faces, the params dispatch advances DEST by the face stride (16 physical rows) using `TTI_SETRWC` (Wormhole) or `inc_dst_addr<8>()` twice (Blackhole).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, approximation mode, and include macro for HARDSIGMOID
   **Key Findings**: `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func()` returns `{"hardsigmoid_tile_init();", "hardsigmoid_tile({idst});"}` with no template parameters. `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). `get_macro_definition()` returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (default).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: API header that exposes `hardsigmoid_tile()` and `hardsigmoid_tile_init()`
   **Key Findings**: `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`. `hardsigmoid_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()`. `APPROX` is a JIT-generated `constexpr bool` derived from `math_approx_mode`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU kernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, ITERATIONS>, dst_index, VectorMode::RC)`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Implements piecewise linear hardsigmoid: `result = x * (1/6) + 0.5`, clamped to [0, 1]. Uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`). 8 iterations per face, processing 32 elements per iteration. `APPROXIMATION_MODE` template parameter is accepted but not used in any branch -- the code path is identical for both values.

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Verify Blackhole implementation matches Wormhole
   **Key Findings**: Identical implementation to Wormhole B0.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates face iteration and DEST addressing
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` sets DEST write address, stalls SFPU, loops over 4 faces in `VectorMode::RC`, calls SFPU function once per face, advances DEST between faces with `TTI_SETRWC`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()` sets `ADDR_MOD_7` with all-zero increments. No special-case branch applies to `hardsigmoid`.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Verify Blackhole init matches Wormhole
   **Key Findings**: Same `ADDR_MOD_7` configuration. Blackhole uses `_llk_math_eltwise_unary_sfpu_start_` and `_llk_math_eltwise_unary_sfpu_done_` instead of inline `TTI_STALLWAIT`/`clear` calls.

9. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` constexpr is generated
   **Key Findings**: `APPROX` is a JIT-generated `constexpr bool` written to a generated header: `constexpr bool APPROX = {math_approx_mode};`. Its value comes from `desc.get_hlk_math_approx_mode()`, which is the `math_approx_mode` field from the compute kernel config.

10. **File**: `.claude/references/sfpu-hardware-model.md` (main repo)
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and addressing
    **Key Findings**: Confirmed stride-2 addressing model, ITERATIONS=8 per face, SFPMAD semantics for float add/multiply, CC stack mechanism for v_if/v_endif, Fixed Const 2 = 1.0.
