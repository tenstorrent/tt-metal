## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- TANHSHRINK uses a **dedicated compute kernel** (`tanhshrink_kernel.cpp`) rather than the standard `eltwise_sfpu.cpp` with `SFPU_OP_CHAIN_0`. The kernel directly invokes `tanh_tile(0)` followed by `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)`.

**Important note on dispatch status**: The TANHSHRINK operation is registered in `unary_op_types.hpp` (enum value `TANHSHRINK`) and has a `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` macro in `unary.hpp`. However, the dispatch infrastructure in `unary_op_utils.cpp` is incomplete:
- `get_compute_kernel_path(TANHSHRINK)` falls through to `default: return "eltwise_sfpu.cpp"` -- it does NOT return the dedicated `tanhshrink_kernel.cpp` path.
- `get_op_init_and_func_default(TANHSHRINK)` falls through to `default: TT_THROW("unexpected op type")` -- calling `ttnn::tanhshrink()` at runtime would crash during program creation when `get_block_defines()` attempts to generate the `SFPU_OP_CHAIN_0` defines.

The dedicated kernel file exists but is orphaned from the dispatch path. On this branch, TANHSHRINK is non-functional.

**Important note on SFPU implementation status**: The core SFPU implementation for `tanh` (`ckernel_sfpu_tanh.h`) and its LLK dispatch header (`llk_math_eltwise_unary_sfpu_tanh.h`) have been deleted from this branch as part of the deep nuke operation (see `DEEP_NUKE_MANIFEST.md`, Phase 1). The `tanh_tile()` API function in `compute_kernel_api.h` still references `llk_math_eltwise_unary_sfpu_tanh<>()`, but this function has no implementation. Even if the dispatch path were fixed, the kernel would fail to compile due to the missing tanh SFPU symbol.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | N/A for SFPU_OP_CHAIN (dedicated kernel). The compute kernel calls `tanh_tile_init()` and `tanh_tile(0)` with default template argument `fast_and_approx = false`. |
| Effective SFPU path | Would use the non-approximate tanh path (accurate mode) | `template <bool fast_and_approx = false> ALWI void tanh_tile(uint32_t idst)` in `compute_kernel_api.h` line 177-178. The `fast_and_approx=false` value would propagate to `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>()`. |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **Compute Kernel** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` |
| **API Header (tanh)** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180: `tanh_tile_init()`, `tanh_tile()`) |
| **API Header (binary_dest_reuse)** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 208-257: `binary_dest_reuse_tiles_init()`, `binary_dest_reuse_tiles()`) |
| **LLK Dispatch (tanh)** | **DELETED** -- `llk_math_eltwise_unary_sfpu_tanh.h` was nuked in Phase 1 deep nuke |
| **Core SFPU Implementation (tanh)** | **DELETED** -- `ckernel_sfpu_tanh.h` was nuked in Phase 1 deep nuke (originally in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/`) |
| **Parameters Dispatch (SFPU generic)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole B0) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain
The tanhshrink compute kernel follows a two-phase execution model: first an SFPU phase (tanh), then an FPU phase (subtraction).

**Phase 1 -- SFPU tanh**:
1. `tanhshrink_kernel.cpp` calls `tanh_tile_init()` (line 31) which expands to `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` via the `MATH()` macro on the math RISC-V thread.
2. `tanhshrink_kernel.cpp` calls `tanh_tile(0)` (line 32) which expands to `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` on the math thread.
3. The LLK function would invoke `_llk_math_eltwise_unary_sfpu_params_<false>()` with the tanh SFPU functor, iterating over all 4 faces in `VectorMode::RC` mode with 8 iterations per face.
4. The SFPU functor (`_calculate_tanh_<false>` from `ckernel_sfpu_tanh.h`) would execute on each face -- **but this function is deleted on this branch**.

**Phase 2 -- FPU subtraction**:
5. `tanhshrink_kernel.cpp` calls `binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)` (line 34). This configures the unpack and math units for element-wise subtraction where the DEST register content (now holding `tanh(x)`) will be moved to SRCB.
6. `tanhshrink_kernel.cpp` calls `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` (lines 35-36). This unpacks the original input tile `x` from `cb_input` into SRCA, moves `tanh(x)` from DEST into SRCB, and computes `SRCA - SRCB = x - tanh(x)` on the FPU. The result is written back to DEST register index 0.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces processed). The standard tanh operation would process all faces of the tile since tanhshrink operates element-wise on the full tile.
- **Operation invocation**: The tanh SFPU functor would be called once per face (4 times total) from the `_llk_math_eltwise_unary_sfpu_params_` dispatch loop. Each invocation processes 8 iterations (ITERATIONS=8) covering one 16x16 face.
- **DEST address progression**: Standard DEST progression. On **Wormhole B0**, the params dispatch uses raw `TTI_SETRWC` instructions to advance the DEST write pointer by 8+8=16 physical rows (1 face stride) between faces. Within each face, the tanh SFPU functor would use `dst_reg++` to advance 1 sfpi row per iteration (8 iterations = 8 sfpi rows = 16 physical rows = full face). On **Blackhole**, the params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (equivalent to the same face stride).
- **Address mode**: `ADDR_MOD_7` configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- the default SFPU address mode that does not conflict with A2D (which uses ADDR_MOD_0 and ADDR_MOD_2). This is the same on both Wormhole B0 and Blackhole. No special address mode is configured for the tanh SfpuType since it would not match any of the constexpr branches in `eltwise_unary_sfpu_configure_addrmod()`.

### Annotated Compute Kernel Source

Since the core tanh SFPU implementation is deleted, we present the compute kernel itself (which IS the operation's unique logic) and the surviving API stubs.

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
    init_sfpu(cb_input, cb_output); // Configures unpack/math/pack for SFPU mode

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1); // Wait for input tile
            tile_regs_acquire();        // Acquire DEST register

            // Copy input tile x from CB to DEST[0]
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);  // Unpack tile 0 from cb_input to DEST[0]

            // Phase 1: SFPU -- compute tanh(x) in-place in DEST[0]
            tanh_tile_init();           // Init SFPU for tanh (fast_and_approx=false by default)
            tanh_tile(0);              // DEST[0] = tanh(DEST[0]) -- SFPU operation on all 4 faces

            // Phase 2: FPU -- compute x - tanh(x)
            // DEST_TO_SRCB: move DEST[0] (tanh(x)) to SRCB register
            // ELWSUB: compute SRCA - SRCB
            // SRCA gets fresh unpack of cb_input tile 0 (original x)
            // Result: x - tanh(x) written back to DEST[0]
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_input);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_input, 0, 0);       // FPU subtraction: x - tanh(x)

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_output);    // Pack DEST[0] to output CB
            tile_regs_release();
            cb_pop_front(cb_input, 1);  // Pop consumed input tile
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
```

### API Stubs (Surviving References)

The `tanh_tile_init()` and `tanh_tile()` API functions survive in the compute kernel API header, even though the LLK functions they call are deleted:

```cpp
// File: tt_metal/hw/inc/api/compute/compute_kernel_api.h (lines 154-180)

template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<fast_and_approx, DST_ACCUM_MODE>()));
    // llk_math_eltwise_unary_sfpu_tanh_init is DELETED -- would fail to link
}

template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)));
    // llk_math_eltwise_unary_sfpu_tanh is DELETED -- would fail to link
}
```

### SFPU Instructions Used

Due to the nuked state of the tanh SFPU implementation, the actual SFPU instructions cannot be verified in the current codebase. Based on the SFPU hardware model reference and the surviving `sinh` implementation (which uses the same `exp_21f` family), the tanh implementation would have used:

| Instruction | Purpose | Verified in Current Codebase? |
|-------------|---------|------------------------------|
| `SFPLOAD` | Load elements from DEST rows into LREGs for SFPU processing | No (tanh kernel deleted) |
| `SFPSTORE` | Store LREG results back to DEST rows | No (tanh kernel deleted) |
| `SFPMAD` | Fused multiply-add for polynomial approximation or exp composition | No (tanh kernel deleted) |
| `SFPLOADI` | Load immediate constants (e.g., log2(e), thresholds) | No (tanh kernel deleted) |
| `SFPSETCC` / `SFPENCC` | Condition code manipulation for clamping and range checks | No (tanh kernel deleted) |
| `SFPNONLINEAR` (InstrMod=5) | Hardware-accelerated tanh approximation (Quasar only) | Yes -- verified in `ckernel_instr_params.h` for Quasar; NOT available on Wormhole B0 or Blackhole |

For the **FPU subtraction phase** (which IS functional):

| Instruction/Operation | Purpose | Verified? |
|-----------------------|---------|-----------|
| `llk_unpack_A<NONE, true, DEST_TO_SRCB>` | Unpack input tile to SRCA, move DEST content to SRCB | Yes -- `eltwise_binary.h` line 249 |
| `llk_math_eltwise_binary<ELWSUB, NONE, DST_ACCUM_MODE, LoFi, DEST_TO_SRCB>` | FPU element-wise subtraction SRCA - SRCB | Yes -- `eltwise_binary.h` line 254 |

### SFPU Register Usage

**SFPU Phase (tanh -- nuked)**:
- **DEST register**: DEST[0] (tile index 0) -- receives the input tile via `copy_tile()`, then tanh SFPU operates in-place on all 4 faces
- **LREGs**: Usage cannot be determined (implementation deleted). Typical tanh implementations use LREG0 for loading DEST data, LREG1-LREG3 for intermediate computation, and programmable constant registers for polynomial coefficients.

**FPU Phase (subtraction -- functional)**:
- **DEST register**: DEST[0] -- contains tanh(x) before subtraction; receives result x - tanh(x) after
- **SRCB register**: Receives tanh(x) from DEST via `DEST_TO_SRCB` reuse path
- **SRCA register**: Receives original input x via fresh unpack from `cb_input`

**Circular buffers**:
- `c_0` (cb_input): Input tile buffer (2 tiles)
- `c_2` (cb_output): Output tile buffer (2 tiles)
- No intermediate CB (`c_1`) is used by tanhshrink

### Address Mode Configuration

**SFPU phase (tanh)**:
The tanh SfpuType does not exist in the current `SfpuType` enum (only `unused`, `frac`, `swish`, `atanh`, `sinh` are defined in `llk_sfpu_types.h`). Had it existed, the init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::tanh>()` would configure:

- **ADDR_MOD_7**: `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- the default SFPU address mode, used by all SFPU operations that do not match a special-case `constexpr if` branch in `eltwise_unary_sfpu_configure_addrmod()`.
- This is identical on both **Wormhole B0** and **Blackhole**.
- Within the SFPU kernel, the `dst_reg++` operator in the iteration loop handles DEST address advancement (1 sfpi row = 2 physical DEST rows per iteration).

**FPU phase (binary_dest_reuse)**:
The FPU subtraction uses the standard binary eltwise address modes configured by `llk_math_eltwise_binary_init<ELWSUB, NONE, LoFi, DEST_TO_SRCB>()`. These are FPU-level address modes (ADDR_MOD_0/ADDR_MOD_2 for A2D), not SFPU address modes.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Primary compute kernel for tanhshrink -- the main subject of analysis.
   **Key Findings**: Composite operation using tanh_tile(0) SFPU call + binary_dest_reuse_tiles<ELWSUB,DEST_TO_SRCB> FPU subtraction. Self-contained kernel, does NOT use SFPU_OP_CHAIN_0.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Contains dispatch logic (get_compute_kernel_path, get_op_init_and_func_default, get_op_approx_mode).
   **Key Findings**: TANHSHRINK has no case in any switch statement -- falls to default in all three functions. get_compute_kernel_path returns "eltwise_sfpu.cpp" (wrong), get_op_init_and_func_default would TT_THROW, get_op_approx_mode returns false.

3. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Contains tanh_tile_init() and tanh_tile() API function templates.
   **Key Findings**: API stubs survive (template defaults fast_and_approx=false), but they call llk_math_eltwise_unary_sfpu_tanh which has no definition.

4. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Contains binary_dest_reuse_tiles_init() and binary_dest_reuse_tiles() used for the x - tanh(x) subtraction.
   **Key Findings**: DEST_TO_SRCB moves DEST tile to SRCB, then ELWSUB computes SRCA - SRCB. For non-MUL operations, uses MathFidelity::LoFi.

5. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Documents which SFPU implementations were deleted.
   **Key Findings**: ckernel_sfpu_tanh.h (wh+bh+quasar) was deleted in Phase 1 as part of Family 1 (Exponential-Composition) primitives. Originally contained _calculate_tanh_, sigmoid via tanh.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Standard SFPU params dispatch layer -- would be used by tanh LLK.
   **Key Findings**: VectorMode::RC iterates 4 faces with TTI_SETRWC face advancement. Standard pattern for all unary SFPU ops.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base SFPU init and address mode configuration.
   **Key Findings**: ADDR_MOD_7 configured with all-zero increments as default. SfpuType enum only has {unused, frac, swish, atanh, sinh} -- no tanh entry.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU instruction semantics and hardware model reference.
   **Key Findings**: SFPNONLINEAR InstrMod=5 provides hardware-accelerated tanh (Quasar only). On WH/BH, tanh was implemented via software (exp-based polynomial or LOADMACRO, now deleted).

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Surviving exp-composition family member -- shows the exp_21f helper that tanh would have used.
   **Key Findings**: sinh uses exp_21f(z_pos) and exp_21f(z_neg) to compute (exp(x)-exp(-x))/2. The tanh implementation likely used a similar approach: tanh(x) = (exp(2x)-1)/(exp(2x)+1) or 2*sigmoid(2x)-1.
