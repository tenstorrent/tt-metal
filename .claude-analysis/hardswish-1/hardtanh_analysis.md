## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile(0, 0x________u, 0x________u)` (where the two hex literals are the IEEE 754 bitcast of `min_val` and `max_val` respectively, defaulting to `-1.0f` and `1.0f`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit case for HARDTANH) |
| Template parameter (SFPU_OP_CHAIN) | none (not parameterized by approx) | `get_op_init_and_func()` -- the parameterized case emits `hardtanh_tile_init()` with no template argument; `hardtanh_tile(idst, param0, param1)` also has no approx template argument in its expansion |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed through `APPROX` macro to `calculate_hardtanh<false, 8>`. The kernel does not branch on `APPROXIMATION_MODE` -- it is unused in the function body. | The `calculate_hardtanh` template accepts `APPROXIMATION_MODE` but never references it in any `if constexpr` branch. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** calls `hardtanh_tile(idst, param0, param1)` (the `SFPU_OP_CHAIN_0` macro expansion).
2. **API Header** (`hardtanh.h`) wraps this as `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardtanh.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardtanh<APPROXIMATE, 8>, dst_index, VectorMode::RC, param0, param1)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then in `VectorMode::RC` mode loops over 4 faces, calling `calculate_hardtanh(param0, param1)` once per face with `SETRWC` between faces to advance the DEST write pointer.
5. **Core SFPU Implementation** (`ckernel_sfpu_hardtanh.h`) executes the clamping logic: load each SFPU row from DEST, clamp to `[min_val, max_val]`, write back.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed).
- **Operation invocation**: The params dispatch calls `calculate_hardtanh<APPROXIMATE, 8>(param0, param1)` once per face in a `for (int face = 0; face < 4; face++)` loop. Each call processes 8 SFPU iterations (= 1 face = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical rows = 1 face). On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which does `math::inc_dst_addr<8>()` twice (equivalent effect). The address mode configured during init is `ADDR_MOD_7` with `dest.incr = 0` (both WH and BH), meaning the `SETRWC`-based inter-face advance is the only DEST pointer progression between faces; within a face, `dst_reg++` in the SFPI kernel handles per-iteration advancement.

### Annotated SFPU Kernel Source

The kernel uses **Style A: SFPI-based** abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`).

The Wormhole B0 and Blackhole implementations are **identical** -- only one copy is shown.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0); // SFPLOADI: load immediate float from bitcast uint32
    sfpi::vFloat max_val = Converter::as_float(param1); // SFPLOADI: load immediate float from bitcast uint32

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        v_if(val < min_val) { val = min_val; } // SFPSETCC (LT0 after subtract) + CC-guarded SFPMOV
        v_endif; // SFPENCC to restore all-lanes-enabled

        v_if(val > max_val) { val = max_val; } // SFPSETCC (GT after subtract) + CC-guarded SFPMOV
        v_endif; // SFPENCC to restore all-lanes-enabled

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Detailed instruction mapping for `v_if(val < min_val)`:**
The SFPI `operator<(vFloat, vFloat)` comparison generates a subtract (via `SFPMAD`: `val - min_val` or equivalent) followed by `SFPSETCC` testing the sign of the result. If `val < min_val`, the result is negative, and CC.Res is set for those lanes. The `v_if` block pushes CC state, and the assignment `val = min_val` executes only on lanes where the condition holds (CC-guarded `SFPMOV`). The `v_endif` pops the CC stack and restores all-lanes-enabled via `SFPENCC`/`SFPPOPC`.

The same pattern applies for `v_if(val > max_val)` with the comparison direction reversed.

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| `SFPLOADI` | Load the immediate float constants `min_val` and `max_val` into LREGs from their bitcast uint32 representations. Two SFPLOADI per constant (16-bit halves) to construct a full 32-bit float. |
| `SFPLOAD` | Load 32 elements from the current DEST row pair into an LREG (`dst_reg[0]`). Executed once per iteration. |
| `SFPMAD` | Used implicitly by the `<` and `>` comparisons between `vFloat` operands. The comparison `val < min_val` computes `val - min_val` via `SFPMAD(val, 1.0, -min_val)` to produce a result whose sign determines the comparison outcome. |
| `SFPSETCC` | Set per-lane condition codes based on the sign of the subtraction result. Mode `LREG_LT0` (for `<`) or `LREG_GTE0` (for `>`, via complement). |
| `SFPPUSHC` | Push current CC state onto the CC stack at the start of each `v_if` block to enable nested/sequential conditional regions. |
| `SFPMOV` | CC-guarded move of `min_val` or `max_val` into the `val` register for lanes where the clamp condition is true. |
| `SFPPOPC` | Pop CC state from the stack at `v_endif` to restore the pre-condition CC state. |
| `SFPENCC` | Enable/disable condition code masking. Used by `v_endif` to return to all-lanes-enabled state after each conditional block. |
| `SFPSTORE` | Write the clamped result back to the current DEST row pair. Executed once per iteration. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (min_val)** | Holds the `min_val` constant loaded from `param0` via `Converter::as_float`. Persists across all 8 iterations (loaded once before the loop). |
| **LREG (max_val)** | Holds the `max_val` constant loaded from `param1` via `Converter::as_float`. Persists across all 8 iterations (loaded once before the loop). |
| **LREG (val)** | Temporary register holding the current DEST element value. Loaded from `dst_reg[0]` each iteration, potentially overwritten by `min_val` or `max_val` during clamping, then stored back. |
| **LREG (temp)** | Temporary for the subtraction result used by `vFloat < vFloat` / `vFloat > vFloat` comparisons. Holds `val - min_val` or `val - max_val` for `SFPSETCC` testing. |
| **DEST rows** | Input/output tile data. Each iteration reads and writes one sfpi row (= 2 physical DEST rows = 32 elements). Over 8 iterations, one face (256 elements) is processed. Over 4 face invocations, the full tile (1024 elements) is processed. |
| **CC register** | Per-lane condition code bits. Set by `SFPSETCC` after each comparison, used to guard the clamping `SFPMOV`, then cleared by `SFPENCC`/`SFPPOPC` at `v_endif`. |
| **CC stack** | Used by `v_if`/`v_endif` to save and restore CC state across sequential conditional blocks. Each `v_if` pushes, each `v_endif` pops. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardtanh>()`.

Since `SfpuType::hardtanh` does not match any of the special-case `if constexpr` branches (which check for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is set:

| Field | Value | Meaning |
|-------|-------|---------|
| `srca.incr` | 0 | No auto-increment for SrcA |
| `srcb.incr` | 0 | No auto-increment for SrcB |
| `dest.incr` | 0 | No auto-increment for DEST |

This is **identical for both Wormhole B0 and Blackhole**. The `ADDR_MOD_7` with `dest.incr = 0` means the SFPU does not auto-increment the DEST pointer between instruction executions; instead, within-face progression is handled by `dst_reg++` in the SFPI code (which directly manipulates the DEST read/write counter), and between-face progression is handled by `SETRWC` instructions emitted by the params dispatch layer.

No additional `ADDR_MOD_6` is configured for this operation (that path requires `typecast`/`unary_max`/`unary_min` types).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 macro expansion, approximation mode, and parameter encoding for HARDTANH.
   **Key Findings**: HARDTANH uses `eltwise_sfpu.cpp`, macro `SFPU_OP_HARDTANH_INCLUDE`, init `hardtanh_tile_init()`, tile func `hardtanh_tile(idst, param0, param1)` with `param0`/`param1` as IEEE 754 bitcast uint32. `get_op_approx_mode` returns `false` (default).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Confirm HARDTANH is a parameterized type and understand the dispatch template.
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`. Two parameters: `min_val` (default -1.0) and `max_val` (default 1.0).

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
   **Reason**: Trace the API-level tile function signature and its delegation to the LLK layer.
   **Key Findings**: `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)` delegates to `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`. `hardtanh_tile_init()` delegates to `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
   **Reason**: Trace the LLK dispatch layer that bridges the API to the core SFPU function.
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardtanh<APPROXIMATE, 8>, dst_index, VectorMode::RC, param0, param1)`. Init delegates to `llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>()`.

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Read the core SFPU kernel implementation.
   **Key Findings**: Simple clamping kernel using SFPI abstractions. Loads `min_val` and `max_val` from bitcast uint32 params, then iterates 8 times per face: load from DEST, clamp with two `v_if` blocks, store back. `APPROXIMATION_MODE` template parameter is accepted but unused.

6. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Verify Blackhole implementation is identical to Wormhole.
   **Key Findings**: Byte-for-byte identical to the Wormhole version.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch layer (face iteration, DEST addressing, stall logic).
   **Key Findings**: For `VectorMode::RC`, loops 4 faces, calls the SFPU function once per face, uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces to advance DEST pointer by 16 physical rows.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand the init function and address mode configuration.
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardtanh>()` calls `_init_sfpu_config_reg()`, then `eltwise_unary_sfpu_configure_addrmod<hardtanh>()` which sets `ADDR_MOD_7` with all increments = 0. No special-case address mode for hardtanh.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand the `Converter::as_float` utility used in the kernel.
   **Key Findings**: Simple union-based bitcast from `uint32_t` to `float`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU instruction semantics, register model, and addressing.
    **Key Findings**: Confirmed stride-2 model, ITERATIONS=8 per face, CC mechanism for `v_if`/`v_endif`, `SFPMAD` used for float comparisons (subtract), `SFPSETCC` for condition code setting.

11. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Understand how SFPI C++ abstractions (`vFloat` comparisons, `v_if`/`v_endif`) map to hardware instructions.
    **Key Findings**: `operator<(vFloat, vFloat)` creates a `__vCond` with `__vCondLT`, which generates a subtract + `SFPSETCC` sequence. `v_if` pushes CC stack, `v_endif` pops and restores.
