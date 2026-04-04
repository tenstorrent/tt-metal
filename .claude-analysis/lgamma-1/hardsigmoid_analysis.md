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
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` directly) | `get_op_init_and_func_default()` returns `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)` -- no parameterized template argument; the API header passes `APPROX` which resolves to `false` at compile time via JIT `chlkc_descriptors.h` |
| Effective SFPU path | `APPROXIMATION_MODE=false`; the kernel does not branch on `APPROXIMATION_MODE` -- the template parameter is declared but unused in `calculate_hardsigmoid` | The `calculate_hardsigmoid` function body has no `if constexpr (APPROXIMATION_MODE)` branch; both approximate and non-approximate paths execute identical code |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` (identical on both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` (identical on both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. The compute kernel `eltwise_sfpu.cpp` expands `SFPU_OP_CHAIN_0` which calls `hardsigmoid_tile_init()` once and then `hardsigmoid_tile(0)` per tile.
2. `hardsigmoid_tile(idst)` in `hardsigmoid.h` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` guarded by `MATH((...))` so it only runs on the TRISC_MATH processor.
3. `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROXIMATE>(dst_index, vector_mode)` in the LLK dispatch header calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, 8>` as the callable, `dst_index`, and `VectorMode::RC`.
4. `_llk_math_eltwise_unary_sfpu_params_()` in the params dispatch header sets the DEST write address, configures address mode base, stalls SFPU behind MATH, then iterates over 4 faces (for `VectorMode::RC`), calling `calculate_hardsigmoid<false, 8>()` once per face and issuing `TTI_SETRWC` to advance the DEST pointer between faces.
5. `calculate_hardsigmoid<false, 8>()` in `ckernel_sfpu_hardsigmoid.h` is the core SFPU function that processes 8 SFPI iterations (one face worth of data: 8 x 32 = 256 elements).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the 32x32 tile are processed, covering all 1024 elements.
- **Operation invocation**: The params dispatch calls `calculate_hardsigmoid<false, 8>()` once per face in a loop of 4 iterations. Each call processes 8 SFPI rows (one face). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is issued twice (on Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (on Blackhole), advancing the DEST pointer by 16 physical rows to the start of the next face.
- **DEST address progression**: Standard DEST progression. On both Wormhole and Blackhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (no auto-increment from address mode). Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering 8 iterations = 256 elements per face. Between faces, `SETRWC` advances by face stride. `ADDR_MOD_7` is used because it avoids conflicting with `ADDR_MOD_0` and `ADDR_MOD_2` used by the A2D (Accumulate-to-DEST) path.

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
    constexpr float one_sixth = 1.0f / 6.0f; // 0x3E2AAAAB in IEEE 754

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face, processing 32 elements each
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: read 32 elements from current DEST row pair into LREG
        sfpi::vFloat result = x * one_sixth + 0.5f; // SFPMAD: result = x * (1/6) + 0.5 (fused multiply-add)

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; } // SFPSETCC (sign bit test) + CC stack ops; SFPLOADI 0.0 (CC-guarded)
        v_endif;
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // Compare result > 1.0 via subtraction + sign test; SFPMOV from Fixed Const 2 (CC-guarded)
        v_endif;

        sfpi::dst_reg[0] = result; // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Description | Usage in this kernel |
|-------------|-------------|---------------------|
| `SFPLOAD` | Load 32 elements from DEST row pair into an LREG with format conversion | Reads the input `x` from the current DEST position at each iteration (`sfpi::dst_reg[0]` read) |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC` | Computes `x * one_sixth + 0.5f` in a single instruction. The constant `one_sixth` (1/6) and `0.5f` are loaded as immediates into LREGs before the MAD. Since `vFloat + vFloat` also lowers to SFPMAD (a * 1.0 + b), any intermediate additions in the SFPI abstraction also use this instruction |
| `SFPLOADI` | Load 16-bit immediate value into an LREG | Loads the constant `0.0f` for the clamp floor (`result = 0.0f` in the first `v_if` block). Also used to materialize the float constants `one_sixth` and `0.5f` into LREGs for the SFPMAD |
| `SFPSETCC` | Set condition code CC.Res based on comparison | Used by `v_if(result < 0.0f)` (tests sign bit of result, mode `LREG_LT0`) and by `v_if(result > sfpi::vConst1)` (tests sign of `result - 1.0`, mode `LREG_GTE0` complemented) to establish per-lane predicates |
| `SFPENCC` | Enable/disable condition code masking | Used at the start and end of each `v_if`/`v_endif` block to activate and deactivate CC-based lane masking |
| `SFPPUSHC` | Push CC state onto the per-lane CC stack | Saves the current CC state at `v_if` entry so it can be restored at `v_endif` |
| `SFPPOPC` | Pop CC state from the per-lane CC stack | Restores the saved CC state at `v_endif`, re-enabling all lanes for subsequent unconditional computation |
| `SFPCOMPC` | Complement CC.Res for else-branch | May be emitted by the SFPI compiler as part of the CC manipulation sequence within `v_if` blocks |
| `SFPMOV` | Copy one LREG to another | Used to assign `sfpi::vConst1` (Fixed Const 2 = 1.0) to the result LREG in the second clamp block (`result = sfpi::vConst1`), CC-guarded |
| `SFPSTORE` | Store LREG contents back to DEST row pair with format conversion | Writes the clamped result back to the current DEST position at each iteration (`sfpi::dst_reg[0] = result`) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** (general purpose) | Used by the SFPI compiler for intermediate values. `x` is loaded into one LREG via SFPLOAD, `result` occupies another LREG after the SFPMAD computation. The constants `one_sixth` and `0.5f` are materialized into LREGs via SFPLOADI pairs (two SFPLOADI instructions per 32-bit float: one for the upper 16 bits, one for the lower 16 bits). The exact LREG assignment is determined by the SFPI register allocator |
| **Fixed Const 2** (CREG_IDX_1 = index 10) | Holds `1.0f` (reset value `0x3F800000`). Accessed as `sfpi::vConst1` for the upper clamp comparison and assignment (`result > vConst1`, `result = vConst1`). This is a read-only constant register shared across all SFPU slices |
| **DEST register** | Input/output storage. Each iteration reads 32 elements (2 physical DEST rows x 16 elements/row) from the current DEST position and writes the result back to the same position. The DEST pointer advances by 1 sfpi row per iteration via `dst_reg++` |
| **CC register** (per-lane) | Used for predicated execution in the two `v_if` blocks. CC.En is toggled on/off to gate lane execution during clamping. CC.Res is set by SFPSETCC based on sign bit tests of the comparison results |
| **CC stack** (per-lane, 8-entry) | Used by `v_if`/`v_endif` to save and restore CC state. Each `v_if` pushes one entry; each `v_endif` pops one entry. Maximum stack depth in this kernel is 1 (no nested conditionals) |

### Address Mode Configuration

The address mode for hardsigmoid is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()`, called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardsigmoid>()`.

**Configuration (identical on both Wormhole B0 and Blackhole):**

| Address Mode | srca.incr | srcb.incr | dest.incr | Purpose |
|-------------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode for this operation. No auto-increment from the address mode itself -- DEST advancement is handled explicitly by `dst_reg++` (within the face) and `SETRWC` / `inc_dst_addr` (between faces) |

The `SfpuType::hardsigmoid` does not match any of the specialized `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod()` (those are for `topk_local_sort`, `typecast`, `unary_max/min` variants, and on Blackhole additionally `reciprocal`). Therefore, only `ADDR_MOD_7` with all-zero increments is set, which is the standard configuration for SFPU unary operations.

The choice of `ADDR_MOD_7` (rather than lower-numbered address modes) is deliberate to avoid conflicting with `ADDR_MOD_0` and `ADDR_MOD_2`, which are used by the A2D (Accumulate-to-DEST) data movement path that may operate concurrently.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDSIGMOID
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_default()` returns `hardsigmoid_tile_init();` / `hardsigmoid_tile({idst});`. `get_macro_definition()` returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (default).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: API header layer -- trace from tile-level API call to LLK dispatch
   **Key Findings**: `hardsigmoid_tile(idst)` calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)`. `hardsigmoid_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: LLK dispatch layer -- trace from LLK to core SFPU function
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_hardsigmoid<APPROXIMATE, 8>` as the callable and `VectorMode::RC`.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Core SFPU implementation -- the actual SFPU kernel function
   **Key Findings**: SFPI-based kernel using `vFloat`, `dst_reg`, `v_if`/`v_endif`. Computes `x * (1/6) + 0.5` then clamps to [0, 1]. WH and BH implementations are identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer -- understand how the SFPU function is called per face
   **Key Findings**: For VectorMode::RC, loops 4 faces, calling the sfpu_func once per face and issuing TTI_SETRWC twice between faces to advance DEST by 16 physical rows (one face stride).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_()` calls `_init_sfpu_config_reg()`, then `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()` which sets `ADDR_MOD_7` with all-zero increments (no special-case branches for hardsigmoid), then `math::reset_counters()`.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` compile-time constant is set
   **Key Findings**: `constexpr bool APPROX = {}` is emitted into `chlkc_descriptors.h` with the value from `hlk_desc.get_hlk_math_approx_mode()`, which comes from the ComputeConfig's `math_approx_mode` field.

8. **File**: `build_Debug/libexec/tt-metalium/runtime/sfpi/include/sfpi.h` and `sfpi_constants.h`
   **Reason**: Understand SFPI constant register mappings
   **Key Findings**: `vConst1` maps to `CREG_IDX_1 = 10` which is Fixed Const 2 with reset value `0x3F800000 = 1.0f`.
