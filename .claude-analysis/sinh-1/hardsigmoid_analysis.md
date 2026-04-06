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
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized: `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)` with no template arguments exposed in the chain defines |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout the entire SFPU kernel | `APPROX` is generated as `false` in `chlkc_descriptors.h` by `genfiles.cpp`, propagated through `hardsigmoid_tile<APPROX>` -> `llk_math_eltwise_unary_sfpu_hardsigmoid<false>` -> `calculate_hardsigmoid<false>`. However, the kernel's implementation does not branch on `APPROXIMATION_MODE` -- both paths execute the same code. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` (identical on both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` (identical on both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardsigmoid_tile_init(); hardsigmoid_tile(0);`, calling the API-level functions.
2. **API Header** (`hardsigmoid.h`): `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` on the math thread via the `MATH()` macro. `hardsigmoid_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardsigmoid.h`): The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()` which configures ADDR_MOD and resets counters. The tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, ITERATIONS=8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST write addressing for the tile, stalls until SFPU is ready, then loops over 4 faces in `VectorMode::RC`, calling `calculate_hardsigmoid<false, 8>()` once per face and advancing the DEST face address with `SETRWC`/`inc_dst_addr` between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_hardsigmoid.h`): `calculate_hardsigmoid<false, 8>()` executes 8 iterations (one per sfpi row within a face), each processing 32 elements via the SFPI abstraction layer.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed). This is the default for unary operations without explicit vector mode overrides.
- **Operation invocation**: The params dispatch loops over 4 faces, calling `calculate_hardsigmoid<false, 8>()` once per face. Each invocation runs 8 SFPI iterations (the inner loop), processing all 16 rows of a face (2 physical DEST rows per iteration due to stride-2).
- **DEST address progression**: Standard DEST progression. On Wormhole, `set_addr_mod_base()` and `clear_addr_mod_base()` bracket the SFPU work; within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration (ITERATIONS=8 per face). Between faces, `TTI_SETRWC(CR_D, 8, SET_D)` is issued twice (advancing by 16 physical rows = 1 face stride). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h

namespace ckernel::sfpu {

// hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f; // 0.1666666716337204 in FP32

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position
        sfpi::vFloat result = x * one_sixth + 0.5f; // SFPMAD: result = x * (1/6) + 0.5 (single FMA instruction)

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; } // CC: SFPSETCC(LT0) on result; guarded SFPLOADI/SFPMOV to set result=0
        v_endif; // CC: restore CC state via SFPPOPC/SFPENCC
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // CC: subtract vConst1 (1.0) then SFPSETCC(GTE0); guarded SFPMOV from const reg
        v_endif; // CC: restore CC state

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements back to current DEST position
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| `SFPLOAD` | Load 32 elements from DEST register into LREG (`sfpi::dst_reg[0]` read). Executed once per iteration to read input `x`. |
| `SFPMAD` | Fused multiply-add: computes `x * one_sixth + 0.5f` in a single instruction. This is the core linear transform of hardsigmoid. Also used implicitly in the `result > sfpi::vConst1` comparison (subtracting 1.0 to set up the CC test). |
| `SFPLOADI` | Load immediate value into LREG. Used to materialize the float constants `one_sixth` (0x3E2AAAAB) and `0.5f` (0x3F000000) and `0.0f` into LREGs for use in SFPMAD and conditional assignments. |
| `SFPSETCC` | Set condition code based on LREG value. Used in two `v_if` blocks: (1) `result < 0.0f` tests sign bit (mode LT0), (2) `result > vConst1` tests sign of `result - 1.0` (mode LT0 on the difference, then complement). |
| `SFPENCC` | Enable/disable condition code masking. Used by `v_if`/`v_endif` to activate CC-guarded execution and to reset CC state after conditional blocks. |
| `SFPPUSHC` | Push CC state onto the CC stack. Used at the start of each `v_if` block to save the current CC state for restoration by `v_endif`. |
| `SFPPOPC` | Pop CC state from the CC stack. Used by `v_endif` to restore the CC state saved by the corresponding `v_if`. |
| `SFPCOMPC` | Complement CC result bits. May be emitted by the SFPI compiler for the comparison logic, particularly for `result > vConst1` which requires testing the negated condition. |
| `SFPMOV` | Register-to-register move. Used to assign `sfpi::vConst1` (constant register 10, value 1.0) to `result` in the second `v_if` block's body. |
| `SFPSTORE` | Store LREG value back to DEST register. Executed once per iteration to write the computed result back (`sfpi::dst_reg[0] = result`). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General-purpose working registers used by the SFPI compiler to hold intermediate values: input `x`, the computed `result`, and temporary values for comparisons. The exact LREG allocation is determined by the SFPI compiler's register allocator. |
| **LREG4-LREG7** | Available but not explicitly required by this kernel. LREG7 may be used if the compiler needs indirect addressing (unlikely for this simple kernel). |
| **DEST register** | Source and destination for tile data. `SFPLOAD` reads from the current DEST row pair; `SFPSTORE` writes back to it. Addressing advances by `dst_reg++` (stride-2, covering 32 elements per sfpi row). |
| **Constant Register (CREG_IDX_1 = index 10)** | `sfpi::vConst1` = Fixed Const 2 = 1.0f (0x3F800000). Used for the upper-bound clamping comparison and assignment. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()`, called during `llk_math_eltwise_unary_sfpu_init`. Since `SfpuType::hardsigmoid` does not match any of the special-cased `if constexpr` branches (topk_local_sort, typecast, unary_max/min, reciprocal), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

This configuration is **identical on both Wormhole and Blackhole**. The zero-increment dest field means the hardware address mode does not auto-increment the DEST pointer between SFPU instructions. Instead, DEST address progression within a face is handled by the `dst_reg++` SFPI abstraction (which directly manipulates the SFPU's internal DEST address counter), and between faces it is handled by `SETRWC` instructions (Wormhole) or `math::inc_dst_addr<8>()` calls (Blackhole).

Note: On Wormhole, the params dispatch additionally calls `math::set_addr_mod_base()` and `math::clear_addr_mod_base()` around the SFPU work, establishing ADDR_MOD_7 as the base address mode for the duration of the SFPU computation. On Blackhole, these calls are absent from the params dispatch (the base is set during `_llk_math_eltwise_unary_sfpu_start_`).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDSIGMOID
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). `get_op_init_and_func_default()` returns `{"hardsigmoid_tile_init();", "hardsigmoid_tile({idst});"}`. `get_op_approx_mode()` returns `false` (default). Macro define is `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: API header defining `hardsigmoid_tile()` and `hardsigmoid_tile_init()`
   **Key Findings**: `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` via the `MATH()` macro. `hardsigmoid_tile_init()` calls the corresponding init function. `APPROX` is a compile-time constant generated from `math_approx_mode`.

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU function
   **Key Findings**: Both architectures have identical implementations. Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardsigmoid<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Both architectures have identical code. Pure SFPI kernel using `vFloat`, `dst_reg`, `v_if`/`v_endif`. Computes `x * (1/6) + 0.5` then clamps to [0, 1] with two `v_if` conditional blocks. No branching on `APPROXIMATION_MODE`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates per-face SFPU calls
   **Key Findings**: Wormhole version uses `math::set_addr_mod_base()`, `TTI_STALLWAIT`, loops 4 faces for `VectorMode::RC`, calls sfpu_func() per face, advances DEST with `TTI_SETRWC` between faces.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function (Blackhole variant)
   **Key Findings**: Blackhole version calls `_llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(dst_index)` and `_llk_math_eltwise_unary_sfpu_done_()`. Loops 4 faces, calls sfpu_func() per face, advances DEST with `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and ADDR_MOD configuration for SFPU operations
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()` only configures `ADDR_MOD_7` with all increments = 0. No special ADDR_MOD_6 for hardsigmoid. Both architectures identical.

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understanding how `APPROX` compile-time constant is generated
   **Key Findings**: Line 394: `constexpr bool APPROX = {};` is emitted into `chlkc_descriptors.h` from `desc.get_hlk_math_approx_mode()`, which comes from `math_approx_mode` in `ComputeConfig`.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understanding `sfpi::vConst1` mapping
   **Key Findings**: `vConst1` is defined as `vConst<vFloat>(CREG_IDX_1)` where `CREG_IDX_1 = 10`, mapping to Fixed Const 2 = 1.0f (0x3F800000).

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU architecture reference for instruction semantics, register model, and addressing
    **Key Findings**: Confirmed stride-2 model (32 elements per sfpi iteration), SFPMAD as the core FMA instruction (no dedicated float add), SFPSETCC modes for conditional execution, and constant register layout.
