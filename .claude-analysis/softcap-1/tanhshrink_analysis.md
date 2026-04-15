## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the `tanhshrink` unary operation. Tanhshrink computes `tanhshrink(x) = x - tanh(x)`.

**Critical Note on Current Codebase State**: The tanhshrink operation is in a **partially-nuked state**. The `UnaryOpType::TANHSHRINK` enum value and the `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` registration in `unary.hpp` still exist, but the dispatch integration in `unary_op_utils.cpp` has been removed. Calling `tanhshrink` at runtime will hit the `default: TT_THROW("unexpected op type {}", op_type)` path in `get_op_init_and_func_default()`. Two orphaned compute kernel files exist on disk (`tanhshrink_kernel.cpp` and `tanhshrink_sfpu_kernel.cpp`) but are not wired into the program factory. Additionally, the `tanh_tile()` API they call depends on `llk_math_eltwise_unary_sfpu_tanh`, which has been **completely removed** from the codebase (both the LLK dispatch and the core `ckernel_sfpu_tanh.h` implementation were nuked). These kernel files would not compile in the current state.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: Two orphaned kernel files exist:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` (FPU binary subtraction variant)
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp` (SFPU binary subtraction variant)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable -- tanhshrink uses dedicated compute kernels rather than the `eltwise_sfpu.cpp` dispatch. The kernel directly calls `tanh_tile(0)` followed by a subtraction.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- the switch has only `default: return false`, so TANHSHRINK falls through to false |
| Template parameter (tanh_tile) | `false` (default) | Both compute kernels call `tanh_tile_init()` and `tanh_tile(0)` without explicit template arguments, using the default `fast_and_approx = false` from `compute_kernel_api.h:154,177` |
| Effective SFPU path | The tanh step would use the non-approximate path. However, the tanh LLK and ckernel have been nuked, so no path resolves | `ckernel_sfpu_tanh.h` was deleted; `llk_math_eltwise_unary_sfpu_tanh` is undefined |

### SFPU Abstraction Layers

Tanhshrink is a **composite operation** composed of two steps: (1) tanh via the unary SFPU path, and (2) subtraction via either the FPU binary path or the SFPU binary path. The layers for each step are documented separately.

#### Step 1: tanh_tile() -- Unary SFPU Tanh (NUKED)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180) |
| **LLK Dispatch** | NUKED -- `llk_math_eltwise_unary_sfpu_tanh.h` no longer exists |
| **Core SFPU Implementation** | NUKED -- `ckernel_sfpu_tanh.h` no longer exists (was in `tt_llk_{wormhole_b0,blackhole,quasar}/common/inc/sfpu/`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (standard unary SFPU params, would have been used) |

#### Step 2a: binary_dest_reuse_tiles (FPU binary subtraction, in tanhshrink_kernel.cpp)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 211-257) |
| **LLK Dispatch** | FPU binary path (llk_math_eltwise_binary), not SFPU |
| **Core SFPU Implementation** | This level of abstraction doesn't exist -- uses the FPU matrix unit, not the SFPU |
| **Parameters Dispatch** | This level of abstraction doesn't exist -- FPU path |

#### Step 2b: sub_binary_tile (SFPU binary subtraction, in tanhshrink_sfpu_kernel.cpp)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 39-41, 68) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

#### tanhshrink_kernel.cpp (FPU binary variant)
1. `copy_tile(cb_input, 0, 0)` -- copies input tile from CB to DEST register at index 0
2. `tanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [NUKED -- would not resolve]
3. `tanh_tile(0)` -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` [NUKED -- would not resolve]. Computes tanh(x) in-place in DEST tile 0.
4. `binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)` -- initializes FPU binary subtraction. The `DEST_TO_SRCB` flag causes the current DEST contents (tanh(x)) to be moved to SRCB.
5. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` -- unpacks the original input x from cb_input into SRCA, moves tanh(x) from DEST to SRCB, and computes SRCA - SRCB = x - tanh(x) using the FPU. Result written back to DEST.

#### tanhshrink_sfpu_kernel.cpp (SFPU binary variant)
1. `copy_tile(cb_input, 0, 1)` -- copies input tile from CB to DEST register at **tile index 1**
2. `tanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [NUKED]
3. `tanh_tile(1)` -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(1)` [NUKED]. Computes tanh(x) in-place in DEST tile 1.
4. `copy_tile(cb_input, 0, 0)` -- copies the original input x again from CB to DEST register at tile index 0
5. `sub_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::SUB>()` -- initializes SFPU binary init and address mode
6. `sub_binary_tile(0, 1, 0)` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(0, 1, 0)` -> `_llk_math_eltwise_binary_sfpu_params_<APPROX>(_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>, 0, 1, 0, VectorMode::RC)` -- computes DEST[0] - DEST[1] = x - tanh(x) using the SFPU, result in DEST tile 0.

### Parameters Dispatch Summary (SFPU binary subtraction path)

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile
- **Operation invocation**: The params dispatch (`_llk_math_eltwise_binary_sfpu_params_`) loops over 4 faces, calling `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` once per face (8 iterations each = 32 total iterations covering the full tile). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (advancing DEST address by 16 sfpi rows = 1 face stride).
- **DEST address progression**: Uses `ADDR_MOD_7` on both Wormhole and Blackhole, configured with all-zero increments (`srca.incr=0, srcb.incr=0, dest.incr=0`). The address progression within the SFPU kernel itself is handled by `dst_reg++` (advancing 1 sfpi row = 2 physical DEST rows per iteration). Between faces, SETRWC advances the base address.

### Annotated SFPU Kernel Source

The tanh step (`tanh_tile()`) has been nuked and its source is unavailable. Below is the SFPU binary subtraction kernel which performs the `x - tanh(x)` computation, plus the two compute kernel files.

#### Compute Kernel: tanhshrink_kernel.cpp (FPU binary subtraction variant)

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
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);              // x -> DEST[0]

            tanh_tile_init();                        // [NUKED] init tanh SFPU
            tanh_tile(0);                            // [NUKED] DEST[0] = tanh(x)

            // FPU binary sub: moves tanh(x) from DEST to SRCB, unpacks x from CB to SRCA,
            // computes SRCA - SRCB = x - tanh(x), writes result to DEST[0]
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

#### Compute Kernel: tanhshrink_sfpu_kernel.cpp (SFPU binary subtraction variant)

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    init_sfpu(cb_input, cb_output);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 1);              // x -> DEST[1]

            tanh_tile_init();                        // [NUKED] init tanh SFPU
            tanh_tile(1);                            // [NUKED] DEST[1] = tanh(x)

            // Re-copy x to DEST[0] for subtraction
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);              // x -> DEST[0]

            // SFPU binary sub: DEST[0] = DEST[0] - DEST[1] = x - tanh(x)
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

#### Core SFPU Implementation: _calculate_sfpu_binary_ (BinaryOp::SUB branch)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8> // APPROXIMATION_MODE=false, BINOP=BinaryOp::SUB, ITERATIONS=8
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // 32 sfpi rows per tile (64 physical / stride 2)
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load x from DEST tile 0
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load tanh(x) from DEST tile 1
        sfpi::vFloat result                        = 0.0f;

        // For BinaryOp::SUB, this branch is selected at compile time:
        if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1; // x - tanh(x), emits SFPMAD (a * 1.0 + (-b))
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result to DEST tile 0
        sfpi::dst_reg++;                                             // Advance to next sfpi row (2 physical DEST rows)
    }
}
```

### SFPU Instructions Used

The tanhshrink operation is composed of two SFPU phases. Only the subtraction phase has surviving source code.

#### Phase 1: tanh_tile() -- NUKED
The `ckernel_sfpu_tanh.h` implementation has been deleted from the codebase. Based on the SFPU hardware model, the hardware-accelerated path would have used:

| Instruction | Description | Status |
|-------------|-------------|--------|
| `SFPNONLINEAR` (InstrMod=5) | Hardware-accelerated tanh approximation (1 ULP error for FP16_B) | Available in hardware, but ckernel wrapper is nuked |

#### Phase 2: _calculate_sfpu_binary_ (SUB) -- PRESENT
| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[offset]` read | Loads 32 elements from DEST into an LREG. Called twice per iteration: once for `in0` (x, from tile 0) and once for `in1` (tanh(x), from tile 1). The offset `dst_index * dst_tile_size_sfpi` selects the tile within DEST. |
| `SFPMAD` | `in0 - in1` (vFloat subtraction) | Computes `in0 * 1.0 + (-in1)`. The SFPU has no dedicated subtract instruction; subtraction is encoded as an SFPMAD with the addend sign inverted (InstrMod[1]=1). |
| `SFPSTORE` | `sfpi::dst_reg[offset] = result` | Stores the 32-element result vector back to DEST at the output tile position. |

### SFPU Register Usage

#### Phase 2: SFPU binary subtraction
| Register | Usage |
|----------|-------|
| LREG (via `in0`) | Holds 32 elements of x loaded from DEST tile 0 at the current sfpi row |
| LREG (via `in1`) | Holds 32 elements of tanh(x) loaded from DEST tile 1 at the current sfpi row |
| LREG (via `result`) | Holds the subtraction result `x - tanh(x)`, initialized to 0.0f then overwritten |
| DEST tile 0 | Input tile holding x (copied from cb_input). Also serves as the output tile. |
| DEST tile 1 | Holds tanh(x) after `tanh_tile(1)` completes (SFPU kernel variant only) |

**Note on the FPU variant**: In `tanhshrink_kernel.cpp`, only DEST tile 0 is used. The tanh result occupies DEST[0], then `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` moves the tanh result to SRCB (via the FPU's register reuse mechanism), unpacks x from cb_input into SRCA, and performs the subtraction on the FPU matrix unit.

### Address Mode Configuration

#### SFPU binary subtraction path (sub_binary_tile)

The address mode is configured in `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` (from `llk_math_eltwise_binary_sfpu.h`):

**Wormhole B0 and Blackhole** (identical configuration):
```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

All increments are zero because the SFPU binary kernel manages its own DEST addressing via `dst_reg++` in the inner loop and `TTI_SETRWC` between faces. The zero-increment ADDR_MOD_7 avoids conflicting with `ADDR_MOD_0` and `ADDR_MOD_2` which may be used by the A2D (unpack-to-DEST) pipeline that runs concurrently.

The `SfpuType::unused` template argument means the `if constexpr` branch for integer multiply/max/min operations is NOT taken, so `ADDR_MOD_6` is not configured. Only `ADDR_MOD_7` is set.

**Minor arch difference**: On Wormhole, `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_done_` calls `math::clear_addr_mod_base()`. On Blackhole, these calls are absent (the addr_mod base is managed differently).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine dispatch configuration for TANHSHRINK (compute kernel path, init/func macros, approx mode)
   **Key Findings**: TANHSHRINK is not present in `get_op_init_and_func_default()` or `get_op_init_and_func_parameterized()` -- dispatch was removed during nuke. `get_op_approx_mode()` returns false by default. `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` by default.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Read the FPU-variant compute kernel for tanhshrink
   **Key Findings**: Uses `tanh_tile(0)` on DEST[0], then `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` to compute x - tanh(x) via FPU binary subtraction.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
   **Reason**: Read the SFPU-variant compute kernel for tanhshrink
   **Key Findings**: Copies input to DEST[1], applies `tanh_tile(1)`, re-copies input to DEST[0], then uses `sub_binary_tile(0, 1, 0)` for SFPU-based subtraction.

4. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Find tanh_tile API definition and its template parameters
   **Key Findings**: `tanh_tile<false>(idst)` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)`, which is UNDEFINED (nuked).

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Find sub_binary_tile API definition
   **Key Findings**: `sub_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(idst0, idst1, odst)`.

6. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   **Reason**: Trace LLK dispatch for SFPU binary subtraction
   **Key Findings**: Calls `_llk_math_eltwise_binary_sfpu_params_` with `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` as the callable.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Read the core SFPU binary operation implementation
   **Key Findings**: For `BinaryOp::SUB`, computes `result = in0 - in1` using SFPI vFloat subtraction (emits SFPMAD). Reads from two different tile offsets in DEST and writes to a third.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Understand binary SFPU params dispatch (face iteration, SETRWC pattern)
   **Key Findings**: VectorMode::RC loops 4 faces, calls sfpu_func once per face (8 iterations), advances DEST by SETRWC(CR_D, 8) twice between faces.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: Read binary SFPU address mode configuration and start/done functions
   **Key Findings**: Uses ADDR_MOD_7 with all-zero increments. WH calls set_addr_mod_base()/clear_addr_mod_base(); BH does not.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU instruction semantics, addressing model, and hardware-accelerated functions
    **Key Findings**: SFPNONLINEAR InstrMod=5 provides hardware tanh with 1 ULP error. vFloat subtraction emits SFPMAD with inverted addend sign.

11. **File**: `DEEP_NUKE_MANIFEST.md`
    **Reason**: Understand which tanh-related files were nuked
    **Key Findings**: `ckernel_sfpu_tanh.h` (wh+bh+quasar) and `ckernel_sfpu_tanh_derivative.h` (wh+bh) were both deleted. Tanh is classified as Family 2 (Exponential-based).
