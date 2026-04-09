## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: Not applicable -- TANHSHRINK uses a dedicated compute kernel that does NOT go through the standard `eltwise_sfpu.cpp` + `SFPU_OP_CHAIN_0` dispatch. The kernel directly calls `tanh_tile(0)` followed by `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)`.

**Note on nuked state**: The core SFPU implementation for tanh (`ckernel_sfpu_tanh.h`) was deleted in the Phase 1 deep nuke. The API header (`compute_kernel_api.h`) still declares `tanh_tile()` and `tanh_tile_init()`, but the LLK dispatch function (`llk_math_eltwise_unary_sfpu_tanh`) and the ckernel implementation (`_calculate_tanh_`) no longer exist. Additionally, the `get_op_init_and_func_default` switch in `unary_op_utils.cpp` has no case for `TANHSHRINK` (it would `TT_THROW`), confirming that the standard `eltwise_sfpu.cpp` path was never used for this operation.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (tanh_tile) | `false` (default) | The dedicated compute kernel calls `tanh_tile_init()` and `tanh_tile(0)` without explicit template arguments, so `fast_and_approx` defaults to `false` |
| Effective SFPU path | Standard (non-approximate) tanh via `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>` | The API header at `compute_kernel_api.h:178` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)` -- but this LLK function definition is **nuked** |

### SFPU Abstraction Layers

Tanhshrink is a two-phase operation: (1) SFPU tanh via `tanh_tile()`, then (2) FPU subtraction via `binary_dest_reuse_tiles()`. The layers below trace the SFPU (tanh) component only.

| Layer | File Path |
|-------|-----------|
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` |
| **API Header (tanh)** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180) |
| **LLK Dispatch (tanh)** | **NUKED** -- was `llk_math_eltwise_unary_sfpu_tanh.h` (Wormhole) / equivalent for Blackhole |
| **Core SFPU Implementation (tanh)** | **NUKED** -- was `ckernel_sfpu_tanh.h` in both `tt_llk_wormhole_b0/common/inc/sfpu/` and `tt_llk_blackhole/common/inc/sfpu/` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared infrastructure, still exists) |
| **API Header (binary subtract)** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 208-257) |

### Call Chain

The tanhshrink compute kernel executes the following call chain per tile:

1. **`copy_tile(cb_input, 0, 0)`** -- Copies the input tile from circular buffer `c_0` into DEST register at index 0. After this, DEST[0] contains the input `x`.

2. **`tanh_tile_init()`** -- Calls `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` [NUKED]. This would have configured SFPU address modes and initialized any constants needed for tanh computation.

3. **`tanh_tile(0)`** -- Calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` [NUKED]. This would have invoked the core SFPU function `_calculate_tanh_<false, 8>()` via the `_llk_math_eltwise_unary_sfpu_params_` dispatch, processing all 4 faces (VectorMode::RC). After this, DEST[0] contains `tanh(x)` (the input has been overwritten in-place).

4. **`binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)`** -- Configures the FPU for element-wise subtraction. `DEST_TO_SRCB` means the current DEST content (`tanh(x)`) will be moved to SRCB. The unpack init prepares to read from `cb_input` (the original `x`) into SRCA.

5. **`binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)`** -- Executes the FPU subtraction: SRCA - SRCB = `x - tanh(x)`. The original input `x` is unpacked from `cb_input` into SRCA, the DEST content `tanh(x)` is moved to SRCB, and the FPU computes `x - tanh(x)` which is written back to DEST[0]. This is the final tanhshrink result.

### Parameters Dispatch Summary

Since the tanh SFPU kernel is nuked, the parameters dispatch summary describes the **surviving infrastructure** that the tanh LLK dispatch would have used:

- **Vector mode**: `VectorMode::RC` (all 4 faces processed). The standard tanh dispatch would have called `_llk_math_eltwise_unary_sfpu_params_<false>()` with `vector_mode = (int)VectorMode::RC`, iterating over all 4 faces of the tile.
- **Operation invocation**: The params dispatch function (`llk_math_eltwise_unary_sfpu_params.h`) calls the SFPU function once per face in a `for (int face = 0; face < 4; face++)` loop, with `TTI_SETRWC` advancing the DEST pointer by 16 physical rows (2 SETRWC calls of 8 each) between faces.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `ADDR_MOD_7` is configured with `.dest = {.incr = 0}` (the standard no-auto-increment mode for SFPU ops where `dst_reg++` handles progression). On Blackhole, the same `ADDR_MOD_7` configuration is used.
- **FPU subtraction phase**: After the SFPU tanh completes on all 4 faces, the `binary_dest_reuse_tiles` call uses the FPU (not SFPU) pipeline. The unpack moves `x` from `cb_input` to SRCA, the DEST content `tanh(x)` is moved to SRCB, and the FPU's element-wise subtract unit computes `SRCA - SRCB = x - tanh(x)`. This operates on entire tiles through the standard FPU math pipeline with `MathFidelity::LoFi` (since `ELWSUB` is not `ELWMUL`).

### Annotated SFPU Kernel Source

#### Part 1: Dedicated Compute Kernel (tanhshrink_kernel.cpp)

This is the full compute kernel that orchestrates the tanhshrink operation. It is NOT a standard `eltwise_sfpu.cpp` kernel -- it is a self-contained kernel that directly calls the tanh SFPU API and the binary FPU subtraction API.

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
    init_sfpu(cb_input, cb_output);                        // Configure unpack/pack for SFPU path

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);                    // Wait for input tile
            tile_regs_acquire();                           // Acquire DEST registers

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);                     // Copy input x to DEST[0]

            tanh_tile_init();                              // Init SFPU for tanh (NUKED)
            tanh_tile(0);                                  // DEST[0] = tanh(x), in-place (NUKED)

            // FPU subtraction: x - tanh(x)
            // DEST_TO_SRCB: move DEST content (tanh(x)) to SRCB
            // Unpack original x from cb_input into SRCA
            // FPU computes SRCA - SRCB = x - tanh(x)
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);                       // Pack result to output CB
            tile_regs_release();
            cb_pop_front(cb_input, 1);                     // Release input tile
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

#### Part 2: tanh_tile API (surviving declaration)

The API header declarations survive but their LLK implementations are deleted:

```cpp
// File: tt_metal/hw/inc/api/compute/compute_kernel_api.h (lines 154-180)

template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()));  // NUKED
}

template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));  // NUKED
}
```

#### Part 3: Core SFPU Implementation (NUKED)

The file `ckernel_sfpu_tanh.h` was deleted from both `tt_llk_wormhole_b0/common/inc/sfpu/` and `tt_llk_blackhole/common/inc/sfpu/` in the Phase 1 deep nuke. According to the nuke manifest (`DEEP_NUKE_MANIFEST.md`), the deleted function was `_calculate_tanh_`, which computed tanh using sigmoid via the identity `tanh(x) = 2 * sigmoid(2x) - 1`.

The tanh implementation depended on the following primitives, all of which were also nuked:
- `ckernel_sfpu_exp.h` -- exponential (`_sfpu_exp_21f_bf16_`, `_calculate_exponential_*`)
- `ckernel_sfpu_sigmoid.h` -- sigmoid (`_calculate_sigmoid_`)

**No source code is available for the core SFPU tanh implementation.**

### SFPU Instructions Used

Since the core SFPU tanh implementation is nuked, the exact SFPU instructions cannot be verified from source. Based on the surviving hardware model documentation and the nuke manifest:

| Instruction | Description | Used by |
|-------------|-------------|---------|
| `SFPLOAD` | Load data from DEST row into LREG for SFPU processing | tanh SFPU kernel (NUKED) |
| `SFPSTORE` | Store LREG result back to DEST row | tanh SFPU kernel (NUKED) |
| `SFPMAD` | Fused multiply-add, used for polynomial evaluation and arithmetic (e.g., `2*sigmoid(2x) - 1`) | tanh SFPU kernel (NUKED) |
| `SFPLOADI` | Load immediate constants into LREGs | tanh SFPU kernel (NUKED) |
| `SFPSETCC` | Set condition code for predicated execution (clamping, range checks) | tanh SFPU kernel (NUKED) |
| `SFPENCC` | Enable/disable condition code masking | tanh SFPU kernel (NUKED) |
| `SFPEXEXP` | Extract exponent -- used by exp primitive for range reduction | exp helper (NUKED) |
| `SFPSETEXP` | Set exponent -- used by exp primitive for reconstruction | exp helper (NUKED) |
| `SFPDIVP2` | Divide by power of 2 -- exponent manipulation in exp | exp helper (NUKED) |
| `SFPNONLINEAR` (instr_mod=5) | Hardware-accelerated tanh -- **Quasar only**, not available on Wormhole/Blackhole | Quasar alternative path |

**Note**: On Quasar, `SFPNONLINEAR` with `instr_mod1=5` provides a single-instruction hardware-accelerated tanh approximation. On Wormhole and Blackhole, tanh must be synthesized from exp-based primitives via sigmoid.

The FPU subtraction step (`binary_dest_reuse_tiles<ELWSUB>`) does not use SFPU instructions -- it uses the FPU element-wise subtract unit.

### SFPU Register Usage

Since the core SFPU tanh implementation is nuked, register usage cannot be verified from source. Based on the standard patterns observed in surviving SFPU kernels:

| Register | Usage (expected) |
|----------|------------------|
| `dst_reg[0]` / LREG0 | Input value loaded from DEST, holds intermediate and final results |
| LREG1-LREG3 | Intermediate values during exp computation (range reduction, polynomial evaluation) |
| `vConstFloatPrgm0` | Programmable constant (likely `ln(2)` reciprocal or similar for exp) |
| `vConstFloatPrgm1` | Programmable constant (likely polynomial coefficient) |
| DEST[0] | Initially holds input `x` (from `copy_tile`); after `tanh_tile` holds `tanh(x)`; after `binary_dest_reuse_tiles` holds `x - tanh(x)` |
| SRCA | During FPU subtract: original input `x` (unpacked from `cb_input`) |
| SRCB | During FPU subtract: `tanh(x)` (moved from DEST via `DEST_TO_SRCB`) |

### Address Mode Configuration

The address mode configuration for the tanh SFPU operation is determined by the `eltwise_unary_sfpu_configure_addrmod` function in the LLK dispatch layer. Since tanh would use `SfpuType::tanh` (if it existed in the enum), and there is no special-case branch for tanh in the surviving `eltwise_unary_sfpu_configure_addrmod` function, it would fall through to the default configuration:

**Wormhole B0** (`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

**Blackhole** (`tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

Both architectures use the same default `ADDR_MOD_7` with zero increments. DEST address progression is handled explicitly within the SFPU kernel via `dst_reg++` (SFPI abstraction) or manual TTI instructions, not by hardware auto-increment. Between faces, the params dispatch uses `TTI_SETRWC` to advance the DEST pointer by 16 physical rows (2 calls of `inc_dst_addr<8>()`).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Primary compute kernel for the tanhshrink operation
   **Key Findings**: Dedicated kernel that calls `tanh_tile(0)` then `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` to compute `x - tanh(x)`. Does NOT use `SFPU_OP_CHAIN_0` dispatch.

2. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: API header declaring `tanh_tile()` and `tanh_tile_init()`
   **Key Findings**: Template functions with `fast_and_approx = false` default. Call `llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>()` which is NUKED.

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: API for `binary_dest_reuse_tiles` and `binary_dest_reuse_tiles_init` used in the subtraction step
   **Key Findings**: `DEST_TO_SRCB` moves DEST content to SRCB, unpacks from CB into SRCA, FPU computes SRCA - SRCB. Uses `MathFidelity::LoFi` for non-multiply operations.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To check `get_op_approx_mode`, `get_compute_kernel_path`, and `get_op_init_and_func_default` for TANHSHRINK
   **Key Findings**: `get_op_approx_mode` returns `false` (default case). `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` (default case -- but TANHSHRINK actually uses its dedicated kernel). `get_op_init_and_func_default` has no case for TANHSHRINK (would `TT_THROW`), confirming the dedicated kernel path.

5. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: To understand what SFPU primitives were nuked
   **Key Findings**: `ckernel_sfpu_tanh.h` (wh+bh+quasar), `ckernel_sfpu_exp.h`, and `ckernel_sfpu_sigmoid.h` were all deleted in Phase 1. The tanh implementation used sigmoid, which in turn used exp.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: LLK dispatch infrastructure and address mode configuration
   **Key Findings**: Default `ADDR_MOD_7` configuration with zero increments. `_llk_math_eltwise_unary_sfpu_start_` and `_llk_math_eltwise_unary_sfpu_done_` functions for SFPU lifecycle management. `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` advances by 16 physical rows between faces.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function used by all SFPU unary ops
   **Key Findings**: `VectorMode::RC` processes all 4 faces. Each face: call SFPU function once, then 2x SETRWC to advance DEST pointer. This is the standard dispatch path that the nuked tanh LLK dispatch would have used.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware architecture reference
   **Key Findings**: `SFPNONLINEAR` with `instr_mod=5` provides hardware-accelerated tanh on Quasar only. Not available on Wormhole/Blackhole. Stride-2 addressing model. ITERATIONS=8 per face.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_ops.h`
   **Reason**: Verify SFPNONLINEAR instruction availability
   **Key Findings**: `SFPNONLINEAR` opcode 0x99 exists only in Quasar instruction set. Not present in Wormhole or Blackhole instruction definitions.

10. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list.md`
    **Reason**: Understand tanhshrink's routing category
    **Key Findings**: TANHSHRINK is classified as "Mixed routing" with a dedicated `tanhshrink_kernel.cpp`. This confirms it uses a specialized compute kernel rather than the standard `eltwise_sfpu.cpp` + `SFPU_OP_CHAIN_0` path.

11. **File**: `docs/sfpu_operations/key_notes/tanhshrink_key_notes.md`
    **Reason**: Operation formula and parameter documentation
    **Key Findings**: `tanhshrink(x) = x - tanh(x)`. No parameters. Deterministic, mode-independent.
