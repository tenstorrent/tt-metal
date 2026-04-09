## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the `PRELU_SFPU` unary operation.

**NOTE**: This analysis was performed on a deep-nuked repository where the `PRELU_SFPU` operation's source code (dispatch entries in `unary_op_utils.cpp`, API header `prelu.h`, LLK dispatch `llk_math_eltwise_unary_sfpu_prelu.h`, and core SFPU kernel `ckernel_sfpu_prelu.h`) have been removed. The analysis below is **reconstructed** from:
1. Pre-nuke documentation (`docs/sfpu_operations/unary_eltwise_sfpu_list.md`, `prelu_sfpu_key_notes.md`, `prelu_tile.rst`)
2. Surviving structurally-identical operations (threshold, hardtanh, clamp) that demonstrate the same parametrized SFPI-style kernel pattern
3. The surviving dispatch infrastructure (`unary_op_utils.cpp`, `llk_math_eltwise_unary_sfpu_params.h`, `llk_math_eltwise_unary_sfpu_macros.h`)
4. The formula: `max(0, x) + weight * min(0, x)`, equivalent to `x if x >= 0, weight * x if x < 0`

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU` (removed from enum in deep nuke; was present in pre-nuke `unary_op_types.hpp`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `SFPU_OP_INIT_0 SFPU_OP_FUNC_0` where `SFPU_OP_INIT_0` = `prelu_tile_init();` and `SFPU_OP_FUNC_0` = `prelu_tile(0, param0);`
- **Include guard macro**: `SFPU_OP_PRELU_INCLUDE`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `param0` (weight as `uint32_t`) | `get_op_init_and_func_parameterized()` -- parameterized case: `prelu_tile_init()` / `prelu_tile(idst, param0)` where `param0` is the bit-cast weight |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout | The `_calculate_prelu_` function is templated on `APPROXIMATION_MODE` but the prelu algorithm uses simple comparison and multiply -- no approximation-dependent branches |

### SFPU Abstraction Layers
List the file path for each abstraction layer. Since the operation was deep-nuked, the pre-nuke paths are reconstructed from documentation and surviving patterns.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` [REMOVED -- confirmed by Doxyfile:959 and `prelu_tile.rst`] |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_prelu.h` [REMOVED -- reconstructed from surviving pattern in `llk_math_eltwise_unary_sfpu_frac.h`] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_prelu.h` [REMOVED -- reconstructed from formula and surviving `ckernel_sfpu_threshold.h`, `ckernel_sfpu_clamp.h`] |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` [SURVIVING -- shared infrastructure] |

### Call Chain
The SFPU kernel is invoked from the compute kernel through the following chain:

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `prelu_tile_init(); prelu_tile(0, param0);` which calls the API header functions.
2. **API header** (`prelu.h`): `prelu_tile(idst, param0)` calls `MATH((llk_math_eltwise_unary_sfpu_prelu<APPROX>(idst, param0)))` and `prelu_tile_init()` calls `MATH((llk_math_eltwise_unary_sfpu_prelu_init<APPROX>()))`.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_prelu.h`): `llk_math_eltwise_unary_sfpu_prelu<APPROXIMATE>(dst_index, param0)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_prelu_<APPROXIMATE, 8>, dst_index, (int)VectorMode::RC, param0)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, then loops over 4 faces calling `_calculate_prelu_<APPROXIMATE, 8>(param0)` once per face, with `TTI_SETRWC` between faces to advance the DEST write pointer.
5. **Core SFPU function** (`ckernel_sfpu_prelu.h`): `_calculate_prelu_<APPROXIMATION_MODE, ITERATIONS>(param0)` converts `param0` to a `vFloat` weight, then iterates 8 times per face, applying `x if x >= 0, weight * x if x < 0` to each DEST row pair.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (standard for element-wise unary operations).
- **Operation invocation**: The params dispatch function loops over 4 faces, calling `_calculate_prelu_<APPROXIMATE, 8>(param0)` once per face. Each call processes 8 SFPU iterations (= 8 sfpi rows = 256 elements = one full face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces on Wormhole / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole). Address mode `ADDR_MOD_7` is configured with `dest.incr=0` (SFPU uses explicit `dst_reg++` rather than hardware auto-increment for DEST addressing).

### Annotated SFPU Kernel Source

The core SFPU kernel was removed in the deep nuke. The reconstruction below is based on:
- The mathematical formula: `max(0, x) + weight * min(0, x)` = `x if x >= 0, weight * x if x < 0`
- The API signature from `prelu_tile.rst`: `prelu_tile(uint32_t idst, uint32_t param0)` -- one uint32_t parameter (bit-cast weight)
- The `SFPU_OP_PRELU_INCLUDE` macro group (standalone, not shared with other ops)
- Surviving reference implementations: `ckernel_sfpu_threshold.h` (single-param conditional), `ckernel_sfpu_clamp.h` (SFPI v_if/v_endif conditional clamping), `ckernel_sfpu_hardtanh.h` (multi-param conditional with `s2vFloat16b`)

The kernel style is **Style A: SFPI-based** -- all surviving relu-family and conditional activation kernels in this codebase (threshold, clamp, hardtanh) use SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_prelu.h
// [RECONSTRUCTED -- original file removed in deep nuke]

#pragma once

#include <cstdint>

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>  // APPROXIMATION_MODE=false
inline void _calculate_prelu_(std::uint32_t param0)
{
    // param0 is the PReLU weight, bit-cast from float to uint32_t
    sfpi::vFloat weight = Converter::as_float(param0);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair

        v_if (in < 0.0f)  // SFPSETCC + CC stack: test sign of input
        {
            in = in * weight;  // SFPMAD: multiply negative values by weight
        }
        v_endif;  // CC stack restore

        sfpi::dst_reg[0] = in;  // SFPSTORE: write result back to DEST

        sfpi::dst_reg++;  // advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}

} // namespace ckernel::sfpu
```

**Reconstruction confidence**: HIGH. The pattern is identical to `leaky_relu` (same formula with `negative_slope` replaced by `weight`) and matches the surviving `threshold`, `clamp`, and `hardtanh` kernels in structure. The API signature from `prelu_tile.rst` confirms a single `uint32_t param0` parameter. The `Converter::as_float` pattern for converting `uint32_t` to float is used by `threshold` and other parametrized kernels. The `v_if (in < 0.0f)` conditional branching pattern is standard for relu-family operations.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|------------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements (2 physical rows x 16 elements/row) from current DEST position into an LREG |
| `SFPSTORE` | `sfpi::dst_reg[0] = in` (write) | Store 32 elements from LREG back to DEST at current position |
| `SFPMAD` | `in * weight` | Fused multiply-add used for `in * weight + 0.0`. Also used implicitly for the scalar `0.0f` comparison setup |
| `SFPSETCC` | `v_if (in < 0.0f)` | Set per-lane condition code: CC.Res = 1 if `in < 0`, enabling predicated execution on negative lanes |
| `SFPENCC` | `v_if` / `v_endif` | Enable/disable condition code masking. `v_if` enables CC, `v_endif` disables it (all lanes active) |
| `SFPPUSHC` | `v_if` (implicit) | Push current CC state onto per-lane CC stack to support the conditional block |
| `SFPPOPC` | `v_endif` (implicit) | Pop CC state from stack, restoring pre-conditional state |

### SFPU Register Usage

| Register | Usage | Description |
|----------|-------|-------------|
| **LREG0** | Input/output | Holds the current DEST row pair data during processing. Loaded via `SFPLOAD`, modified conditionally, stored via `SFPSTORE` |
| **LREG (weight)** | Scalar broadcast | The `weight` parameter, converted from `uint32_t` to `vFloat` via `Converter::as_float`. Broadcast to all 32 lanes as a scalar constant. Compiler allocates this to an available LREG (typically LREG1 or LREG2) |
| **DEST rows** | Source/destination | Each iteration accesses 2 physical DEST rows (32 elements) via the stride-2 addressing. 8 iterations per face, 4 faces per tile = 32 iterations total covering all 1024 elements |
| **CC bits** | Per-lane predication | Used by `v_if (in < 0.0f)` to mask the multiply operation. Only lanes where `in < 0` execute the `in * weight` instruction |

### Address Mode Configuration

The address mode configuration for `PRELU_SFPU` uses the standard unary SFPU setup from `eltwise_unary_sfpu_configure_addrmod<SfpuType>()`:

**Wormhole B0 and Blackhole (identical for this operation):**

| ADDR_MOD | Field | Value | Description |
|----------|-------|-------|-------------|
| `ADDR_MOD_7` | `srca.incr` | 0 | No auto-increment for SrcA (not used by SFPU) |
| `ADDR_MOD_7` | `srcb.incr` | 0 | No auto-increment for SrcB (not used by SFPU) |
| `ADDR_MOD_7` | `dest.incr` | 0 | No hardware auto-increment for DEST -- SFPU kernel uses explicit `dst_reg++` for row advancement |

`PRELU_SFPU` does not match any of the special-case `SfpuType` values that configure `ADDR_MOD_6` with `dest.incr=2` or `dest.incr=32` (those are for `typecast`, `unary_max/min`, `signbit`, `topk_local_sort`). It uses only `ADDR_MOD_7` with all-zero increments, relying on the SFPI `dst_reg++` abstraction for DEST pointer advancement.

## Local Knowledge Sources
### Local References
1. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list.md`
   **Reason**: Pre-nuke catalog of all SFPU operations with their macro groups and parametrization status
   **Key Findings**: PRELU_SFPU uses `SFPU_OP_PRELU_INCLUDE` as a standalone macro group, is parametrized with a weight param

2. **File**: `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md`
   **Reason**: Operation-specific notes documenting formula and parameters
   **Key Findings**: Formula is `max(0, x) + weight * min(0, x)`, weight default init = 0.25

3. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/prelu_tile.rst`
   **Reason**: Doxygen API documentation confirming function signatures
   **Key Findings**: API signature is `prelu_tile(uint32_t idst, uint32_t param0)` and `prelu_tile_init()`

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Surviving reference implementation of a single-param conditional SFPI kernel
   **Key Findings**: Shows the `Converter::as_float(param)` pattern for uint32_t-to-vFloat conversion, `v_if`/`v_endif` conditional execution, `dst_reg[0]` read/write, `dst_reg++` iteration

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Surviving reference implementation of a multi-param conditional SFPI kernel
   **Key Findings**: Shows `s2vFloat16b(param)` alternative conversion, `v_if`/`v_endif` branching with value replacement

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Surviving reference implementation showing v_if/v_elseif/v_endif pattern
   **Key Findings**: Demonstrates multi-branch conditional with `v_elseif`, parameter conversion via `s2vFloat16a`

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Shared parameters dispatch infrastructure for all unary SFPU operations
   **Key Findings**: VectorMode::RC loops over 4 faces, calling the SFPU function once per face with TTI_SETRWC between faces

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base LLK infrastructure for unary SFPU (init, start, done, address mode configuration)
   **Key Findings**: `ADDR_MOD_7` with `dest.incr=0` is the standard configuration for most unary SFPU ops. Special ADDR_MOD_6 only for typecast/max/min/signbit/topk

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
   **Reason**: Surviving per-operation LLK dispatch file showing the concrete pattern
   **Key Findings**: Confirms the pattern: includes init.h and params.h, defines init function calling `llk_math_eltwise_unary_sfpu_init<SfpuType, APPROXIMATE>()`, defines tile function calling `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_fn<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, ...params)`

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
    **Reason**: Macro library providing standardized dispatch patterns
    **Key Findings**: `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro matches prelu's pattern (one runtime uint32_t param). `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT` variant uses `<sfpi::vFloat, APPROXIMATE, 8, uint32_t>` template args

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware reference for instruction semantics, register layout, and addressing model
    **Key Findings**: Stride-2 addressing (dst_reg++ = 2 physical rows = 32 elements), ITERATIONS=8 per face, SFPMAD used for all float arithmetic, SFPSETCC/SFPENCC for CC management

12. **File**: `.claude-analysis/rrelu-1/reference_selection.md`
    **Reason**: Reference selection document explaining why prelu_sfpu was chosen as a reference for rrelu
    **Key Findings**: Confirms prelu_sfpu formula, standalone SFPU_OP_PRELU_INCLUDE pattern, and parameter-as-float convention
