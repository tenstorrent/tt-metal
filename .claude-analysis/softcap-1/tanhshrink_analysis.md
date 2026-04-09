## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp` (primary, uses FPU binary subtraction); also `tanhshrink_sfpu_kernel.cpp` (SFPU-only variant using SFPU binary subtraction)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable. TANHSHRINK uses a **dedicated compute kernel** rather than the generic `eltwise_sfpu.cpp` dispatch. In the non-nuked codebase, `get_compute_kernel_path(UnaryOpType::TANHSHRINK)` returns `"tanhshrink_kernel.cpp"`. The kernel directly calls `tanh_tile(0)` followed by `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` instead of relying on the `SFPU_OP_CHAIN_0` macro mechanism.

**Note on codebase state**: This is a "deeply nuked" evaluation branch. The `ckernel_sfpu_tanh.h` SFPU implementation (containing `_calculate_tanh_`) and its LLK dispatch (`llk_math_eltwise_unary_sfpu_tanh.h`) were deleted in Phase 1 of the deep nuke. The compute kernel files (`tanhshrink_kernel.cpp`, `tanhshrink_sfpu_kernel.cpp`) survive, as does the `tanh_tile()` API declaration in `compute_kernel_api.h`, but the underlying LLK/ckernel implementation that `tanh_tile()` dispatches to is absent. The SFPU binary subtraction path (`ckernel_sfpu_binary.h`) survives intact.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::TANHSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (tanh_tile) | `false` (default) | `tanh_tile<false>(0)` -- the compute kernel calls `tanh_tile(0)` without a template argument, so `fast_and_approx` defaults to `false` per the API declaration `template <bool fast_and_approx = false>` |
| Effective SFPU path | Standard (non-approximate) tanh computation | The `false` template argument propagates through `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>` to the core SFPU function `_calculate_tanh_<false>` |

### SFPU Abstraction Layers

TANHSHRINK is a **composite operation** that chains two distinct operations in its dedicated compute kernel: (1) SFPU tanh and (2) FPU or SFPU binary subtraction. The abstraction layers are documented for each component.

#### Component 1: tanh (SFPU)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180: `tanh_tile_init<>()`, `tanh_tile<>()`) |
| **LLK Dispatch** | [NUKED] Was `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` -- deleted in deep nuke Phase 1 |
| **Core SFPU Implementation** | [NUKED] Was `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_tanh.h` -- deleted in deep nuke Phase 1 |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared by all unary SFPU ops) |

#### Component 2a: FPU binary subtraction (primary kernel)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 211-257: `binary_dest_reuse_tiles_init<>()`, `binary_dest_reuse_tiles<>()`) |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API directly calls `llk_unpack_A` + `llk_math_eltwise_binary` |
| **Core SFPU Implementation** | This level of abstraction doesn't exist -- FPU binary subtraction uses the Matrix Unit (FPU), not the SFPU |
| **Parameters Dispatch** | This level of abstraction doesn't exist -- FPU binary ops are configured by the math init functions |

#### Component 2b: SFPU binary subtraction (SFPU variant kernel)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 39, 68: `sub_binary_tile()`, `sub_binary_tile_init()`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

**Primary kernel (`tanhshrink_kernel.cpp`):**

1. `tanh_tile_init()` -> `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()` -> [NUKED: would initialize tanh SFPU state]
2. `tanh_tile(0)` -> `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)` -> [NUKED: would call `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::_calculate_tanh_<false>, 0, VectorMode::RC)` which iterates over 4 faces]
3. `binary_dest_reuse_tiles_init<ELWSUB, DEST_TO_SRCB>(cb_input)` -> configures unpack A and FPU binary math for subtraction with DEST-to-SRCB reuse
4. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` -> unpacks input tile from `cb_input` to SRCA, moves DST[0] (containing tanh result) to SRCB, then computes SRCA - SRCB = x - tanh(x), writing result to DST[0]

**SFPU variant kernel (`tanhshrink_sfpu_kernel.cpp`):**

1. `copy_tile(cb_input, 0, 1)` -> copies input tile from `cb_input` into DST[1]
2. `tanh_tile_init()` + `tanh_tile(1)` -> computes tanh in-place on DST[1], so DST[1] = tanh(x)
3. `copy_tile(cb_input, 0, 0)` -> copies input tile again from `cb_input` into DST[0], so DST[0] = x
4. `sub_binary_tile_init()` + `sub_binary_tile(0, 1, 0)` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(0, 1, 0)` -> `_llk_math_eltwise_binary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>, 0, 1, 0, VectorMode::RC)` -> iterates over 4 faces, computing DST[0] - DST[1] = x - tanh(x)

### Parameters Dispatch Summary

**For the tanh SFPU component (Component 1):**

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed
- **Operation invocation**: The core `_calculate_tanh_` function is called once per face (4 times total), with each call processing ITERATIONS=8 sfpi rows per face
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `ADDR_MOD_7` is set (dest.incr=0, no auto-increment) since the SFPI `dst_reg++` handles address progression explicitly. `SETRWC(CR_D, 8)` is called twice between faces to advance by 16 physical DEST rows (= 1 face).

**For the SFPU binary subtraction (Component 2b):**

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed
- **Operation invocation**: `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` is called once per face (4 times total), taking three tile indices (in0, in1, out). Each call iterates 8 times, reading from both input tiles and writing to the output tile in DEST.
- **DEST address progression**: Standard DEST progression with binary addressing. Each iteration reads from `dst_reg[dst_index_in0 * 32]` and `dst_reg[dst_index_in1 * 32]`, writes to `dst_reg[dst_index_out * 32]`, then `dst_reg++` advances all three pointers. `SETRWC(CR_D, 8)` is called twice between faces. `ADDR_MOD_7` is set (dest.incr=0).

### Annotated SFPU Kernel Source

Since the core tanh SFPU implementation (`ckernel_sfpu_tanh.h`) was deleted in the deep nuke, only the SFPU binary subtraction kernel is available. The two compute kernels themselves are included first, followed by the surviving SFPU binary subtraction implementation.

#### Primary Compute Kernel

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

            // Copy input tile x from cb_input into DST[0]
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);

            // Compute tanh(x) in-place in DST[0] using SFPU
            tanh_tile_init();
            tanh_tile(0); // DST[0] = tanh(x)

            // Compute x - tanh(x) using FPU binary subtraction with DEST_TO_SRCB reuse:
            // - DST[0] (= tanh(x)) is moved to SRCB
            // - Input tile x is unpacked from cb_input into SRCA
            // - FPU computes SRCA - SRCB = x - tanh(x), result goes to DST[0]
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

#### SFPU Variant Compute Kernel

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

            // Copy input tile into DST[1] for tanh computation
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 1); // DST[1] = x

            // Compute tanh in-place on DST[1]
            tanh_tile_init();
            tanh_tile(1); // DST[1] = tanh(x)

            // Copy input tile again into DST[0] (need original x for subtraction)
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0); // DST[0] = x

            // SFPU binary subtraction: DST[0] - DST[1] = x - tanh(x), result in DST[0]
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

#### SFPU Binary Subtraction Core Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

// POW/DIV/XLOGY implementations removed -- depend on exp/log/recip primitives
// Generator must implement from SFPI instructions

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // APPROXIMATION_MODE unused for SUB, BINOP=BinaryOp::SUB, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // sfpi rows per tile (64 physical / stride 2)
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DST[in0] at current row
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DST[in1] at current row
        sfpi::vFloat result = 0.0f;

        // For BinaryOp::SUB: compiles to SFPMAD(in0, 1.0, -in1) or equivalent
        if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1;
        }
        // ... other BinaryOp cases omitted for brevity (ADD, MUL, RSUB, etc.)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE result to DST[out] at current row
        sfpi::dst_reg++; // advance all dst_reg pointers by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() {} // No-op init for binary SUB
```

### SFPU Instructions Used

The tanhshrink operation uses instructions from two distinct SFPU operations:

#### From tanh_tile (Component 1) -- [NUKED, inferred from API documentation]

The tanh SFPU implementation was deleted in the deep nuke. Based on the DEEP_NUKE_MANIFEST.md, `ckernel_sfpu_tanh.h` contained `_calculate_tanh_` which computed tanh, typically via a sigmoid-based identity: `tanh(x) = 2 * sigmoid(2x) - 1`. The exact SFPU instructions used cannot be verified from the current codebase.

#### From SFPU binary subtraction (Component 2b) -- verified

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Loads 32 elements (2 physical DEST rows) from a DEST tile into an SFPU LREG. Used to read `in0` and `in1` values via `dst_reg[index * 32]`. |
| `SFPMAD` | Multiply-accumulate: `a * b + c`. Used to implement `in0 - in1` as `in0 * 1.0 + (-in1)`. There is no dedicated float subtraction instruction; subtraction is expressed through SFPMAD with appropriate sign manipulation. |
| `SFPSTORE` | Stores 32 elements from an SFPU LREG back to a DEST tile. Used to write the subtraction result via `dst_reg[index * 32] = result`. |

#### From FPU binary subtraction (Component 2a) -- not SFPU

The primary kernel's `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` uses the FPU (Matrix Unit) for subtraction, not the SFPU. The FPU binary subtraction operates on full tiles via the SRCA/SRCB/DEST register pipeline. This is not documented in the SFPU instructions table as it is an FPU operation.

### SFPU Register Usage

#### tanh_tile (Component 1) -- [NUKED, inferred]

- **DEST register**: DST[0] is used as both input and output -- tanh is computed in-place
- **Programmable constants**: The tanh implementation likely used `vConstFloatPrgm0/1/2` for polynomial coefficients (common pattern for exponential/sigmoid-based approximations)
- **LREGs**: Standard SFPU lane registers (LREG0-LREG3) would be used for intermediate computation

#### SFPU binary subtraction (Component 2b) -- verified

- **DEST register**: Three tile slots are referenced:
  - `DST[dst_index_in0 * 32]` (tile 0): source operand 1 (x in tanhshrink)
  - `DST[dst_index_in1 * 32]` (tile 1): source operand 2 (tanh(x) in tanhshrink)
  - `DST[dst_index_out * 32]` (tile 0): output destination (x - tanh(x))
  - Note: in0 and out share the same tile index (0), so the result overwrites the first input
- **LREGs**: LREG0 and LREG1 are implicitly used by `SFPLOAD` to hold `in0` and `in1`, and the result LREG is written back via `SFPSTORE`
- **`dst_reg` progression**: Incremented once per iteration via `dst_reg++`, advancing 1 sfpi row = 2 physical DEST rows = 32 elements per iteration

#### FPU binary subtraction (Component 2a) -- primary kernel

- **SRCA register**: Holds the input tile x (unpacked from `cb_input`)
- **SRCB register**: Holds the tanh result (moved from DEST[0] via `DEST_TO_SRCB` reuse mechanism)
- **DEST register**: DST[0] receives the FPU subtraction result (SRCA - SRCB = x - tanh(x))

### Address Mode Configuration

#### tanh_tile (unary SFPU)

For unary SFPU operations on Wormhole, the address mode is configured in `eltwise_unary_sfpu_configure_addrmod` (in `llk_math_eltwise_unary_sfpu.h`):

- **ADDR_MOD_7**: `{.srca.incr = 0, .srcb.incr = 0, .dest.incr = 0}` -- the base address mode for all unary SFPU ops. The SFPU handles address progression internally via `dst_reg++` in the SFPI abstraction (or equivalent raw instruction addressing). No hardware auto-increment of the DEST address is needed.

Wormhole and Blackhole share the same address mode configuration for standard unary SFPU ops. The `SETRWC(CR_D, 8)` instruction (called twice between faces) advances the DEST write pointer by 16 physical rows to skip to the next face.

#### SFPU binary subtraction

For binary SFPU operations on Wormhole, the address mode is configured in `eltwise_binary_sfpu_configure_addrmod` (in `llk_math_eltwise_binary_sfpu.h`):

- **ADDR_MOD_7**: `{.srca.incr = 0, .srcb.incr = 0, .dest.incr = 0}` -- same as unary SFPU. The `_calculate_sfpu_binary_` function handles address progression explicitly via `dst_reg++`.
- **ADDR_MOD_6** (conditional): Only set for specific ops (mul_int32, max, min, etc.) with `dest.incr = 2`. NOT set for `BinaryOp::SUB` since `SfpuType::unused` is passed as the template parameter, which does not match any of the conditional cases.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To find `get_compute_kernel_path()`, `get_op_approx_mode()`, and `get_op_init_and_func_default()` for TANHSHRINK dispatch
   **Key Findings**: TANHSHRINK is not present in any switch statement -- dispatch was nuked. `get_op_approx_mode()` has only `default: return false`. `get_compute_kernel_path()` has only `default: return "eltwise_sfpu.cpp"`.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
   **Reason**: Primary compute kernel for tanhshrink
   **Key Findings**: Uses `tanh_tile(0)` + `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` -- SFPU tanh followed by FPU subtraction

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
   **Reason**: SFPU-only variant compute kernel
   **Key Findings**: Copies input to DST[1], applies tanh_tile(1), copies input again to DST[0], then uses sub_binary_tile(0, 1, 0) for pure SFPU subtraction

4. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: API declarations for `tanh_tile<>()` and `tanh_tile_init<>()`
   **Key Findings**: Both are templated with `bool fast_and_approx = false`. `tanh_tile()` calls `llk_math_eltwise_unary_sfpu_tanh<fast_and_approx, DST_ACCUM_MODE>(idst)`.

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: API for `binary_dest_reuse_tiles<>()` used in primary tanhshrink kernel
   **Key Findings**: With `DEST_TO_SRCB`, the DST tile is moved to SRCB, input is unpacked to SRCA, and FPU computes the binary operation on SRCA/SRCB

6. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: API for `sub_binary_tile()` used in SFPU variant kernel
   **Key Findings**: Calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(idst0, idst1, odst)`

7. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   **Reason**: LLK dispatch for SFPU binary operations
   **Key Findings**: Calls `_llk_math_eltwise_binary_sfpu_params_` with `ckernel::sfpu::calculate_sfpu_binary` -- note naming discrepancy with `_calculate_sfpu_binary_` in the ckernel file

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation for binary operations (SUB, ADD, etc.)
   **Key Findings**: `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` iterates 8 times per face, loading from two source tiles and storing subtraction result to output tile using SFPI abstractions

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch for unary SFPU operations (face iteration pattern)
   **Key Findings**: VectorMode::RC loops over 4 faces, calling sfpu_func once per face, with SETRWC(CR_D, 8) x2 between faces

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
    **Reason**: Parameters dispatch for binary SFPU operations
    **Key Findings**: Same VectorMode::RC pattern as unary, but passes three dst indices (in0, in1, out) to the sfpu_func

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Address mode configuration for unary SFPU ops
    **Key Findings**: ADDR_MOD_7 set to dest.incr=0 for all unary SFPU ops

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
    **Reason**: Address mode configuration for binary SFPU ops
    **Key Findings**: ADDR_MOD_7 set to dest.incr=0 (same as unary); ADDR_MOD_6 with dest.incr=2 only for specific int/max/min ops, not for SUB

13. **File**: `DEEP_NUKE_MANIFEST.md`
    **Reason**: Understanding which components were removed from the codebase
    **Key Findings**: Phase 1 nuked `ckernel_sfpu_tanh.h` (containing `_calculate_tanh_`, sigmoid via tanh), the LLK dispatch for tanh, and the exponential/sigmoid family primitives it depends on

14. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for tile geometry, addressing, and instruction semantics
    **Key Findings**: Confirmed stride-2 addressing, 32 sfpi rows per tile, 8 iterations per face, SFPMAD for float addition/subtraction
