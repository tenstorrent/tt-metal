## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**Critical Note: Deep-Nuked Codebase**

This analysis is performed on a deep-nuked evaluation branch where the core SFPU tanh implementation (`ckernel_sfpu_tanh.h`) and its LLK dispatch layer (`llk_math_eltwise_unary_sfpu_tanh.h`) have been **deleted** from the codebase. The dedicated compute kernel `tanhshrink_kernel.cpp` still exists and references the deleted `tanh_tile()` API, making the operation **non-functional** at compile time. Additionally, `get_compute_kernel_path()` in `unary_op_utils.cpp` has no case for `TANHSHRINK` -- it falls through to `default: return "eltwise_sfpu.cpp"`, but `get_op_init_and_func_default()` also has no case for `TANHSHRINK` and would throw `TT_THROW("unexpected op type")`. This means the TANHSHRINK operation is broken at both the dispatch routing level and the SFPU implementation level.

This analysis documents: (1) the compute kernel architecture that still exists, (2) the intended SFPU call chain based on API-level declarations, and (3) the mathematical approach for tanhshrink.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` (dedicated kernel, does NOT use the standard `eltwise_sfpu.cpp` + `SFPU_OP_CHAIN_0` dispatch)
- **SFPU_OP_CHAIN_0 expansion**: N/A -- this operation uses a dedicated compute kernel that directly calls `tanh_tile(0)` followed by `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)`. It does not use the `SFPU_OP_CHAIN` mechanism.

**Dispatch Routing Issue**: The current `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` for all ops (including TANHSHRINK) via `default`. However, the actual dedicated kernel is at `tanhshrink_kernel.cpp`. In a functional codebase, this function would have a `case UnaryOpType::TANHSHRINK: return "tanhshrink_kernel.cpp";` entry. The routing has been removed.

#### Mathematical Definition

`tanhshrink(x) = x - tanh(x)`

This is implemented as a two-phase computation:
1. **Phase 1 (SFPU)**: Compute `tanh(x)` in-place on the DEST register using the SFPU
2. **Phase 2 (FPU)**: Compute `x - tanh(x)` using the FPU binary subtract, where the original input `x` is re-read from the circular buffer

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | N/A -- TANHSHRINK uses a dedicated kernel, not `SFPU_OP_CHAIN`. The `tanh_tile<fast_and_approx>()` template defaults to `fast_and_approx = false` (see `compute_kernel_api.h:177`) |
| Effective SFPU path | Default (non-approximate) tanh | The dedicated kernel calls `tanh_tile_init()` and `tanh_tile(0)` without template arguments, so `fast_and_approx` defaults to `false`. The LLK init function `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` would be called, selecting the non-approximate code path |

### SFPU Abstraction Layers
List of file paths for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (declares `tanh_tile_init<>()` and `tanh_tile<>()` at lines 154-180) |
| **LLK Dispatch** | **DELETED** -- would have been `llk_math_eltwise_unary_sfpu_tanh.h` (similar to `llk_math_eltwise_unary_sfpu_sinh.h` pattern). The function `llk_math_eltwise_unary_sfpu_tanh<APPROXIMATE, DST_ACCUM_MODE>(idst)` is referenced but undefined. |
| **Core SFPU Implementation** | **DELETED** -- would have been `ckernel_sfpu_tanh.h` in `tt_llk_wormhole_b0/common/inc/sfpu/` (confirmed in `DEEP_NUKE_MANIFEST.md` line 40: "`ckernel_sfpu_tanh.h` (wh+bh+quasar) -- `_calculate_tanh_`, sigmoid via tanh") |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared params dispatch, still exists) |
| **Binary Subtract API** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (declares `binary_dest_reuse_tiles_init<>()` and `binary_dest_reuse_tiles<>()` at lines 211-257) |

### Call Chain
The tanhshrink compute kernel uses a hybrid SFPU + FPU approach. The intended call chain is:

1. **`tanhshrink_kernel.cpp::kernel_main()`** -- the dedicated compute kernel iterates over tiles
2. **`copy_tile(cb_input, 0, 0)`** -- copies a tile from the input circular buffer to DEST[0]
3. **`tanh_tile_init()`** calls `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [DELETED] -- would initialize SFPU for tanh operation
4. **`tanh_tile(0)`** calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` [DELETED] -- would dispatch to `_llk_math_eltwise_unary_sfpu_params_<false>(calculate_tanh<false, 8>, 0, VectorMode::RC)`, which processes all 4 faces of the tile. DEST[0] now contains `tanh(x)`.
5. **`binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)`** -- initializes the FPU binary engine for subtraction with DEST-to-SRCB reuse
6. **`binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)`** -- moves DEST[0] (containing `tanh(x)`) to SRCB, unpacks the original input `x` from `cb_input` into SRCA, then the FPU computes `SRCA - SRCB = x - tanh(x)` and writes the result back to DEST[0]
7. **`pack_tile(0, cb_output)`** -- packs the final result from DEST to the output circular buffer

### Parameters Dispatch Summary

The tanh SFPU function would have been dispatched through `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`), following the same pattern as other unary SFPU operations like sinh:

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed
- **Operation invocation**: The params dispatch calls the SFPU function once per face (4 times for RC mode), with `TTI_SETRWC` to advance the DEST write pointer between faces. Each function invocation internally iterates `ITERATIONS=8` times (one per sfpi row within a face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the standard `ADDR_MOD_7` is configured with `.dest = {.incr = 0}` (no auto-increment from addr_mod; the SFPU kernel itself handles `dst_reg++`).

### Annotated SFPU Kernel Source

**Note**: The core SFPU tanh implementation (`ckernel_sfpu_tanh.h`) has been **deleted** from this codebase as part of the deep nuke. What follows is the **compute kernel** source that still exists, which shows how tanh is combined with binary subtract to implement tanhshrink.

#### Compute Kernel Source (tanhshrink_kernel.cpp)

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);             // Initialize SFPU pipeline (unpack + pack config)

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);         // Wait for input tile from reader
            tile_regs_acquire();                // Acquire DEST register

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);          // Copy input tile to DEST[0]: DEST = x

            tanh_tile_init();                   // Init SFPU for tanh (calls deleted LLK)
            tanh_tile(0);                       // DEST[0] = tanh(x) via SFPU (calls deleted LLK)

            // FPU binary subtract: x - tanh(x)
            // DEST_TO_SRCB means DEST[0] (tanh(x)) is moved to SRCB,
            // then input tile x is unpacked from cb_input into SRCA,
            // FPU computes SRCA - SRCB = x - tanh(x) and writes back to DEST[0]
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);            // Pack result to output CB
            tile_regs_release();
            cb_pop_front(cb_input, 1);          // Release input tile
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

#### API-Level tanh_tile Declarations (still present)

```cpp
// File: tt_metal/hw/inc/api/compute/compute_kernel_api.h

template <bool fast_and_approx = false>     // fast_and_approx=false by default
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()));
}

template <bool fast_and_approx = false>     // fast_and_approx=false by default
ALWI void tanh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));
}
```

#### Intended LLK Dispatch Pattern (DELETED -- reconstructed from analogous sinh)

Based on the existing `llk_math_eltwise_unary_sfpu_sinh.h` pattern, the deleted tanh LLK would have had this structure:

```cpp
// File: [DELETED] llk_math_eltwise_unary_sfpu_tanh.h (reconstructed pattern)

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tanh.h"              // DELETED -- contained _calculate_tanh_

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_tanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### Core SFPU Implementation (DELETED)

The `ckernel_sfpu_tanh.h` file has been deleted. According to `DEEP_NUKE_MANIFEST.md`, it contained `_calculate_tanh_` and was described as "sigmoid via tanh", indicating it was part of the exponential-composition family. On Wormhole/Blackhole hardware (where `SFPNONLINEAR` is not available), tanh is typically computed via the identity `tanh(x) = 2*sigmoid(2x) - 1` or via `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`, both requiring the exponential primitive which is also deleted.

### SFPU Instructions Used

Because the core SFPU tanh implementation is deleted, the exact instruction list cannot be verified. However, based on:
1. The `DEEP_NUKE_MANIFEST.md` description: "sigmoid via tanh" in the exponential-composition family
2. The sinh kernel (similar family) which uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`)
3. The hardware capabilities (no `SFPNONLINEAR` on Wormhole/Blackhole)

The tanh SFPU kernel would have used these instructions:

| Instruction | Purpose |
|-------------|---------|
| `SFPLOAD` | Load elements from DEST rows into LREGs for processing |
| `SFPMAD` | Fused multiply-add for polynomial evaluation and arithmetic (vFloat operations compile to SFPMAD) |
| `SFPSTORE` | Store computed results back to DEST rows |
| `SFPLOADI` | Load immediate constants (thresholds, polynomial coefficients) |
| `SFPSETCC` | [UNVERIFIED] Set condition codes for range-based branching (small input regime, clamping) |
| `SFPENCC` | [UNVERIFIED] Enable/disable condition code masking for v_if/v_endif blocks |

**Note**: On Quasar hardware, `SFPNONLINEAR` with `TANH_MODE (0x5)` provides a 1-cycle hardware-accelerated tanh approximation (max 1 ULP error in FP16_B). This instruction is NOT available on Wormhole/Blackhole.

Additionally, the tanhshrink kernel uses the **FPU** (not SFPU) for the binary subtract phase:
- `llk_math_eltwise_binary<ELWSUB, NONE, DST_ACCUM_MODE, MathFidelity::LoFi, DEST_TO_SRCB>` -- FPU binary subtract operation

### SFPU Register Usage

**Tanh SFPU Phase** (DELETED -- inferred):
- **DEST[0]**: Initially holds the input tile `x`, transformed in-place to `tanh(x)` by the SFPU tanh kernel. Uses the standard DEST addressing with stride-2 (32 sfpi rows per tile, 8 iterations per face, 4 faces).
- **LREG0-LREG3**: Would have been used as temporaries for intermediate computations (exponential evaluation, polynomial coefficients). The exact allocation is unknown since the implementation is deleted.
- **Programmable constants**: Likely configured via `_init_tanh_()` for exponential-related constants (e.g., ln(2) reciprocal, polynomial coefficients).

**Binary Subtract Phase** (FPU):
- **DEST[0]**: Contains `tanh(x)` from the SFPU phase. Moved to **SRCB** via `DEST_TO_SRCB` reuse mechanism.
- **SRCA**: Receives a fresh copy of the input tile `x` unpacked from `cb_input`.
- **DEST[0]** (output): Receives the FPU subtraction result `SRCA - SRCB = x - tanh(x)`.

### Address Mode Configuration

**SFPU Phase (tanh)**:
- **ADDR_MOD_7** is configured for all standard unary SFPU operations (including tanh) with:
  - `.srca = {.incr = 0}` -- not used by SFPU
  - `.srcb = {.incr = 0}` -- not used by SFPU
  - `.dest = {.incr = 0}` -- no auto-increment; the SFPU kernel manages DEST addressing internally via `dst_reg++`
- This is the same across Wormhole and Blackhole (verified in `llk_math_eltwise_unary_sfpu.h`)
- Between faces, `TTI_SETRWC` advances the DEST write pointer by 16 physical rows (2x `SETRWC(CR_D, 8)` = 16 physical rows = 1 face stride)

**FPU Phase (binary subtract)**:
- The FPU binary subtract uses its own address mode configuration managed by `llk_math_eltwise_binary_init<ELWSUB, ...>`. This is separate from the SFPU address modes and is configured via `binary_dest_reuse_tiles_init`.
- `MathFidelity::LoFi` is used for the subtract operation (sufficient since subtraction is exact in floating point when operands are of similar magnitude).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Primary source -- the dedicated compute kernel for tanhshrink
   **Key Findings**: Implements `x - tanh(x)` by combining SFPU tanh with FPU binary subtract. Uses `DEST_TO_SRCB` reuse to avoid extra data movement.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine dispatch path, approximation mode, and compute kernel routing
   **Key Findings**: `get_op_approx_mode()` returns `false` for all ops (default case). `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` for all ops (default case) -- no explicit case for TANHSHRINK. `get_op_init_and_func_default()` has no case for TANHSHRINK and would throw.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Understand how the compute kernel is selected and configured
   **Key Findings**: Uses `get_compute_kernel_path()` to determine compute kernel path. For TANHSHRINK, this returns `eltwise_sfpu.cpp` (incorrect -- should be `tanhshrink_kernel.cpp`).

4. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Trace the `tanh_tile()` API declaration
   **Key Findings**: `tanh_tile<false>(idst)` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)`. The LLK function is referenced but its definition file has been deleted.

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Understand the `binary_dest_reuse_tiles` API used for the subtraction phase
   **Key Findings**: `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` moves DEST to SRCB, unpacks input to SRCA, then FPU computes SRCA - SRCB. Uses `MathFidelity::LoFi` for non-multiply operations.

6. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Understand what was deleted and why
   **Key Findings**: `ckernel_sfpu_tanh.h` was deleted as part of the Phase 1 exponential-composition family nuke. It contained `_calculate_tanh_` described as "sigmoid via tanh". The exponential primitive (`ckernel_sfpu_exp.h`) was also deleted.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the standard SFPU params dispatch pattern
   **Key Findings**: Standard `_llk_math_eltwise_unary_sfpu_params_` processes 4 faces in RC mode, calls the SFPU function once per face, uses `TTI_SETRWC` to advance between faces.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration and SFPU init pattern
   **Key Findings**: `ADDR_MOD_7` is configured with `dest.incr=0` for all standard unary SFPU ops. `_llk_math_eltwise_unary_sfpu_init_` calls `_init_sfpu_config_reg()` and `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()`.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: Reference for analogous LLK dispatch pattern (sinh is in the same family as tanh)
   **Key Findings**: sinh LLK dispatch follows the pattern: init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`, compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_sinh<...>, dst_index, vector_mode)`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for instruction semantics and addressing
    **Key Findings**: `SFPNONLINEAR` with `InstrMod=5` provides hardware tanh on Quasar only. On Wormhole/Blackhole, tanh must be computed in software. Stride-2 addressing model confirmed. Standard per-face iteration count is 8.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_instr_params.h`
    **Reason**: Verify SFPNONLINEAR tanh mode constant
    **Key Findings**: `p_sfpnonlinear::TANH_MODE = 0x5` confirmed for Quasar. Not present in Wormhole/Blackhole instruction sets.
