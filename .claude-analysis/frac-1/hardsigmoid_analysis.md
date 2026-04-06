## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSIGMOID`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardsigmoid_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSIGMOID)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none (default) | `get_op_init_and_func_default()` -- non-parameterized: `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)` with no explicit template argument; API header passes `APPROX` which resolves to `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_hardsigmoid`; however, the kernel does not branch on `APPROXIMATION_MODE` -- the same linear computation is used regardless | The template parameter is declared but not referenced in any `if constexpr` branch within `calculate_hardsigmoid` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` (identical for WH and BH) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` (identical for WH and BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`hardsigmoid_tile(idst)`** (API header `hardsigmoid.h`): Wraps the call in a `MATH((...))` guard and forwards to `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_hardsigmoid<APPROXIMATE>(dst_index)`** (LLK dispatch): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with the function pointer `ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, 8>`, `dst_index`, and `VectorMode::RC`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (parameters dispatch in tt_llk): Sets the DEST write address for the tile, sets the address mode base, stalls until SFPU is ready, then loops over 4 faces (for `VectorMode::RC`), calling `calculate_hardsigmoid<false, 8>()` once per face, with `TTI_SETRWC` advancing the DEST pointer by 16 physical rows (2 x 8) between faces.
4. **`calculate_hardsigmoid<APPROXIMATION_MODE, ITERATIONS>()`** (core SFPU implementation): Executes the piecewise linear hardsigmoid computation on 8 SFPU iterations (one face).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces processed). The dispatch function iterates `face = 0..3`, calling the SFPU kernel function once per face, which covers all 4 faces of the 32x32 tile.
- **Operation invocation**: For each of the 4 faces, `calculate_hardsigmoid<false, 8>()` is called (ITERATIONS=8). After each face invocation, `TTI_SETRWC` is issued twice with `CR_D, 8` each time, advancing the DEST read/write cursor by 16 physical rows total (= one full face of 16 rows).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces). The address mode used is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}`. The actual per-iteration DEST advancement is handled by the SFPI `dst_reg++` abstraction (stride-2 in hardware), not by the address mode auto-increment.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

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
    constexpr float one_sixth = 1.0f / 6.0f; // 0.16666667f, loaded via SFPLOADI

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];           // SFPLOAD: load 32 elements from current DEST rows into LREG
        sfpi::vFloat result = x * one_sixth + 0.5f;  // SFPMAD(x, one_sixth, 0.5): fused multiply-add, result = x/6 + 0.5

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }       // CC block 1: SFPSETCC(LT0) on result; guarded SFPLOADI/SFPMOV to set result=0.0
        v_endif;                                       // Restore CC to ALL_ENABLED
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // CC block 2: compare result > 1.0 (fixed const); guarded SFPMOV from const reg
        v_endif;                                       // Restore CC to ALL_ENABLED

        sfpi::dst_reg[0] = result;                    // SFPSTORE: write 32 elements back to current DEST rows
        sfpi::dst_reg++;                              // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| `SFPLOAD` | Load 32 elements from the current DEST rows into an LREG. Emitted by `sfpi::vFloat x = sfpi::dst_reg[0]`. |
| `SFPLOADI` | Load 16-bit immediate values into LREGs. Used to materialize the constants `one_sixth` (1/6 = 0.16666667f), `0.5f`, and `0.0f`. Each 32-bit float constant requires two SFPLOADI instructions (high 16 bits + low 16 bits). |
| `SFPMAD` | Fused multiply-add: `result = x * one_sixth + 0.5f`. The core arithmetic of the hardsigmoid linear region. Also used for float addition when the SFPI compiler lowers `vFloat + vFloat` to `SFPMAD(a, 1.0, b)`. |
| `SFPSETCC` | Set condition code based on LREG comparison. Used for `result < 0.0f` (mode `LREG_LT0` -- sign bit test) and `result > vConst1` (implemented as a comparison against the 1.0 constant register). |
| `SFPENCC` | Enable/disable condition code masking. Used by `v_if` to activate CC mode (set CC.En=1) and by `v_endif` to deactivate it (set CC.En=0), restoring all-lanes-active execution. |
| `SFPPUSHC` | Push current CC state onto the CC stack. Part of the `v_if` implementation to save the CC state before the guarded block. |
| `SFPPOPC` | Pop CC state from the CC stack. Part of the `v_endif` implementation to restore the previous CC state. |
| `SFPMOV` | Register-to-register copy. Used within CC-guarded blocks to assign `result = 0.0f` or `result = vConst1` (moving the constant-loaded LREG or const register value to the result LREG, only for lanes where CC is active). |
| `SFPSTORE` | Store 32 elements from an LREG back to the current DEST rows. Emitted by `sfpi::dst_reg[0] = result`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose working registers. Used by the SFPI compiler to hold `x` (loaded from DEST), `result` (the computed hardsigmoid value), and intermediate constant values (`one_sixth`, `0.5f`, `0.0f`). Exact LREG allocation is compiler-determined. |
| **DEST rows** | Input/output: each iteration reads 2 physical DEST rows (32 elements) and writes back the transformed result. The `dst_reg` pointer auto-increments by stride-2 per iteration. |
| **Fixed Const Reg (CREG index 10)** | `sfpi::vConst1` = 1.0f (hardware fixed constant register). Used as the upper clamp value in `v_if(result > sfpi::vConst1)`. |
| **CC register (per-lane)** | Used for predicated execution in the two `v_if` clamping blocks. CC.En is toggled by `SFPENCC`; CC.Res is set by `SFPSETCC` based on the comparison result. |
| **CC stack** | Used by `v_if`/`v_endif` to push/pop CC state (via `SFPPUSHC`/`SFPPOPC`), allowing clean nesting and restoration. |

### Address Mode Configuration

The address mode for this SFPU operation is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()`, called during init.

**Wormhole B0 and Blackhole (identical):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Standard SFPU address mode: no auto-increment on any register. DEST advancement is handled explicitly by the SFPI `dst_reg++` abstraction (which increments the RWC pointer by the stride-2 amount), not by the hardware address mode. |

The `hardsigmoid` SfpuType does not match any of the special-case `if constexpr` branches in the addrmod configuration (those are for `topk_local_sort`, `typecast`, `unary_max/min`, etc.), so only the default `ADDR_MOD_7` with all-zero increments is configured.

Between faces, the params dispatch function advances the DEST cursor via two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls per face, each advancing by 8 physical rows, for a total of 16 physical rows per face transition.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDSIGMOID
   **Key Findings**: `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_default()` returns `hardsigmoid_tile_init();` / `hardsigmoid_tile({idst});`. `get_macro_definition()` returns `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: API header exposing `hardsigmoid_tile()` and `hardsigmoid_tile_init()`
   **Key Findings**: `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`. `hardsigmoid_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_hardsigmoid<APPROXIMATE, 8>` and `VectorMode::RC`. WH and BH files are identical.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Uses pure SFPI abstractions. Computes `x * (1/6) + 0.5`, clamps to [0,1] via two `v_if` blocks. `APPROXIMATION_MODE` template parameter is declared but not referenced in any branch -- same code path regardless. WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that loops over faces and calls the SFPU kernel
   **Key Findings**: For `VectorMode::RC`, loops 4 faces, calling the SFPU function once per face (8 iterations each), with `TTI_SETRWC` advancing DEST by 16 rows between faces.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and addrmod configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()` configures only `ADDR_MOD_7` with `{.dest.incr=0}`. `hardsigmoid` does not match any special-case branches.

7. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model for tile geometry, DEST layout, stride-2 addressing, and instruction semantics
   **Key Findings**: SFPU processes 32 elements per iteration (2 physical DEST rows x 16 elements), 8 iterations per face, 4 faces per tile = 1024 elements. `vFloat + vFloat` emits `SFPMAD`. `SFPLOADI` loads 16-bit immediates. `vConst1` = Fixed Constant register at CREG_IDX_10 = 1.0f.
