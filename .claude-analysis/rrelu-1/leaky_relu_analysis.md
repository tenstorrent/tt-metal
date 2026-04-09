## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**IMPORTANT NOTE**: This analysis is performed on a deep-nuked branch (`vignjatijevic/sfpu-agent-codegen_deeply_nuked_for_rrelu`) where the `LEAKY_RELU` operation has been **fully removed** from all abstraction layers (Phase 2 of the deep nuke). The enum value, TTNN registration, dispatch cases, compute API header, LLK dispatch function, and core SFPU kernel (`_calculate_lrelu_`) are all deleted. The `ckernel_sfpu_relu.h` files for Wormhole, Blackhole, and Quasar contain only `#pragma once`.

This analysis **reconstructs the original architecture** from:
1. The `DEEP_NUKE_MANIFEST.md` which documents exactly what was removed
2. Surviving structurally-identical operations (threshold, sign, fill) that share the same SFPI conditional-branch pattern
3. The surviving LLK dispatch infrastructure (`_llk_math_eltwise_unary_sfpu_params_`, macros, init functions) which is generic and shared by all unary SFPU ops
4. The `reference_selection.md` which describes leaky_relu as structurally identical to rrelu eval mode

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU` [REMOVED from enum in Phase 2 nuke]
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile_init(); leaky_relu_tile(0, param0);` [REMOVED from dispatch]

The original dispatch in `unary_op_utils.cpp` would have had:
- `get_op_init_and_func_parameterized()`: A case for `UnaryOpType::LEAKY_RELU` returning `{"leaky_relu_tile_init();", "leaky_relu_tile(0, param0);"}` where `param0` is the `negative_slope` float parameter bit-cast to `uint32_t`.
- `get_macro_definition()`: Returned `"SFPU_OP_RELU_FAMILY_INCLUDE"` (or `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` if it was part of the default activations header).
- `is_parametrized_type()`: Returned `true` for `LEAKY_RELU`.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- the switch has only a `default: return false` case (no explicit case for LEAKY_RELU even before the nuke) |
| Template parameter (SFPU_OP_CHAIN) | `none` (not parameterized for approx) | `get_op_init_and_func_parameterized()` -- the `param0` was `negative_slope`, not an approximation flag. The `leaky_relu_tile_init()` / `leaky_relu_tile()` calls used default template args which inherit `APPROX` from the compute kernel globals |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `_calculate_lrelu_` | The SFPU kernel's `APPROXIMATION_MODE` template parameter came from the compute API `APPROX` global, which is set by `math_approx_mode`. Since `get_op_approx_mode()` returns `false`, the kernel ran with `APPROXIMATION_MODE=false`. However, leaky_relu's logic (simple conditional multiply) does not have any approximation-dependent branches -- the template parameter is present for API consistency but unused. |

### SFPU Abstraction Layers

All layers listed below have been **deleted** from this branch. The file paths shown are the **original locations** before the nuke, reconstructed from the nuke manifest and the surviving pattern of other ops (frac, swish, atanh, sinh).

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h` [GUTTED -- was the aggregation header that included relu-family compute APIs] |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lrelu.h` [DELETED -- would have contained `llk_math_eltwise_unary_sfpu_lrelu()` and `llk_math_eltwise_unary_sfpu_lrelu_init()`] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_relu.h` [GUTTED -- originally contained `_calculate_lrelu_`] |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` [SURVIVING -- the generic params dispatch is shared by all unary SFPU ops] |

For Quasar, the implementation was in a separate file:
- `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_lrelu.h` [DELETED entirely in Phase 2]

### Call Chain

The original call chain (reconstructed from surviving ops like frac/swish which follow the identical pattern):

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` macro expands to `leaky_relu_tile_init(); leaky_relu_tile(0, param0);`
2. **API Header** (`activations.h` or a dedicated header): `leaky_relu_tile(idst, param0)` calls `MATH((llk_math_eltwise_unary_sfpu_lrelu<APPROX>(idst, param0)))` on the math thread
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_lrelu.h`): `llk_math_eltwise_unary_sfpu_lrelu<APPROXIMATE>(dst_index, param0)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_lrelu_<APPROXIMATE>, dst_index, (int)VectorMode::RC, param0)` -- this is equivalent to the `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro pattern
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, configures ADDR_MOD, stalls for SFPU, then loops over 4 faces calling `_calculate_lrelu_(param0)` per face with `SETRWC` between faces
5. **Core SFPU** (`ckernel_sfpu_relu.h`): `_calculate_lrelu_<APPROXIMATION_MODE, ITERATIONS=8>(uint32_t slope)` performs the element-wise leaky ReLU computation using SFPI abstractions

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces (full tile) are processed. This is the standard mode for element-wise unary operations.
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops `for (int face = 0; face < 4; face++)`, calling `_calculate_lrelu_(param0)` once per face. Each invocation processes `ITERATIONS=8` sfpi rows (= 256 elements = one face). Between faces, two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions advance the DEST write pointer by 16 physical rows (= 1 face stride).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On both Wormhole and Blackhole, `ADDR_MOD_7` is set with `{.dest.incr = 0}` -- this is the standard configuration for most unary SFPU ops. The `lrelu` SfpuType did not appear in the special-case `if constexpr` branches for `ADDR_MOD_6` configuration, meaning it used only the default `ADDR_MOD_7` with zero increment (DEST addressing is managed by `dst_reg++` in the SFPI code, not by hardware auto-increment).

### Annotated SFPU Kernel Source

**CRITICAL: The original kernel source has been deleted.** The `ckernel_sfpu_relu.h` files on this branch contain only `#pragma once`. Below is a **reconstructed** implementation based on:

1. The mathematical definition: `leaky_relu(x) = max(0, x) + negative_slope * min(0, x)` which simplifies to `x if x >= 0, slope * x if x < 0`
2. The surviving structurally-identical kernels (`_calculate_threshold_`, `_calculate_sign_`) which demonstrate the exact SFPI conditional-branch pattern
3. The nuke manifest confirming the original function was named `_calculate_lrelu_` and took a `uint32_t slope` parameter

The kernel used **Style A (SFPI-based)** based on the nuke manifest's categorization of it as part of the "relu family" that used `v_if`/`v_endif` SFPI abstractions.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
// [DELETED from this branch -- reconstructed from architectural patterns]

#include <cstdint>
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_lrelu_(const uint32_t slope) // APPROXIMATION_MODE unused, ITERATIONS=8
{
    // Convert bit-cast uint32_t parameter back to float
    sfpi::vFloat v_slope = Converter::as_float(slope);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position
        v_if (v < 0.0F)                    // SFPSETCC: set condition code for lanes where v < 0
        {
            sfpi::dst_reg[0] = v * v_slope; // SFPMAD: multiply negative values by slope, SFPSTORE: write back
        }
        v_endif;                            // SFPENCC: restore condition codes

        sfpi::dst_reg++;                    // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

} // namespace ckernel::sfpu
```

**Key design notes**:
- Positive values pass through unchanged (identity) because the `v_if` block only executes for negative lanes
- The `slope` parameter arrives as a `uint32_t` bit-pattern of the original `float negative_slope` value, converted back to float via `Converter::as_float()`
- The `APPROXIMATION_MODE` template parameter is present for API consistency but unused -- leaky ReLU has no approximation-dependent logic
- The `#pragma GCC unroll 8` hint matches the standard ITERATIONS=8 loop count, allowing full unrolling for the 8 sfpi rows per face

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler for the `_calculate_lrelu_` kernel:

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[0]` (read) | Loads 32 elements (2 physical DEST rows) from the current DEST position into an SFPU local register (LREG). The stride-2 addressing means each sfpi address spans 2 physical rows. |
| **SFPMAD** | `v * v_slope` | Multiply-accumulate: computes `v * v_slope + 0.0` (addition of zero is implicit). This is the core negative-branch multiplication. SFPMAD is emitted for all vFloat multiplications and additions since there is no dedicated float-add instruction. |
| **SFPSETCC** | `v_if (v < 0.0F)` | Sets the per-lane condition code based on the comparison `v < 0.0`. Lanes where the condition is true have their CC bit set, enabling subsequent predicated execution. |
| **SFPENCC** | `v_endif` | Restores the condition code state, ending the predicated execution region. |
| **SFPSTORE** | `sfpi::dst_reg[0] = ...` (write) | Stores the result from an SFPU local register back to the DEST register at the current position. Only lanes with active CC (inside the `v_if` block) are written. |
| **SFPLOADI** | `Converter::as_float(slope)`, `0.0F` | Loads an immediate constant into an SFPU local register. Used for the slope value and the zero constant in the comparison. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[current]** | Source and destination for the element data. Each iteration reads 32 elements from the current DEST position, conditionally modifies negative values, and writes back. |
| **LREG (local registers)** | Temporary storage for `v` (the loaded input value), `v_slope` (the converted slope parameter), and intermediate multiply results. The SFPI compiler allocates LREGs automatically. Typically LREG0-LREG3 are used. |
| **Condition Code (CC)** | Per-lane 1-bit flags set by `SFPSETCC` (via `v_if`). When CC is set for a lane, that lane's `SFPSTORE` writes are enabled. The CC pattern is simple: set on `v < 0`, used to guard the multiply-and-store, then cleared by `v_endif`. |

### Address Mode Configuration

The address mode for leaky_relu uses the **standard unary SFPU configuration**, identical across Wormhole and Blackhole:

- **ADDR_MOD_7**: `{.srca.incr = 0, .srcb.incr = 0, .dest.incr = 0}` -- configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` in `_llk_math_eltwise_unary_sfpu_init_()`. Since `SfpuType::lrelu` did not appear in any of the special-case `if constexpr` branches (which configure `ADDR_MOD_6` for topk_local_sort, typecast, unary_max/min, signbit), only the default `ADDR_MOD_7` with zero increments was set.

- **DEST address progression**: Managed entirely by SFPI software:
  - Within a face: `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, for 8 iterations
  - Between faces: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice advances 16 physical rows (= 1 face)
  - Total: 4 faces x 8 iterations x 32 elements = 1024 elements = full 32x32 tile

- **No ADDR_MOD_6 specialization**: Unlike typecast/signbit ops, leaky_relu does not need hardware auto-increment of DEST addresses because the SFPI `dst_reg++` idiom handles all address progression within the kernel loop.

## Local Knowledge Sources
### Local References
1. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Authoritative source on what was removed and what survives on this branch
   **Key Findings**: LEAKY_RELU fully removed in Phase 2 (enum, registration, all layers). Original function `_calculate_lrelu_` was in `ckernel_sfpu_relu.h`. The `ckernel_sfpu_lrelu.h` (quasar) was deleted entirely.

2. **File**: `.claude-analysis/rrelu-1/reference_selection.md`
   **Reason**: Provides context on leaky_relu's relationship to rrelu and its structural properties
   **Key Findings**: leaky_relu is structurally identical to rrelu eval mode. Formula: `max(0, x) + negative_slope * min(0, x)`. Uses sign-conditional branching with single float parameter.

3. **File**: `docs/sfpu_operations/key_notes/leaky_relu_key_notes.md`
   **Reason**: Confirms the mathematical formula and parameter conventions
   **Key Findings**: Formula `max(0, x) + negative_slope * min(0, x)`, default slope=0.01, deterministic operation.

4. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list_with_links.md`
   **Reason**: Confirms LEAKY_RELU was a parametrized activation with `negative_slope` parameter
   **Key Findings**: Listed as operation #5, parametrized=Yes, single `negative_slope` param.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Surviving structurally-similar kernel demonstrating the v_if/v_endif conditional pattern with parameter conversion via `Converter::as_float()`
   **Key Findings**: Uses `Converter::as_float(param)` for uint32_t-to-float conversion, `v_if (in <= v_threshold)` for conditional branching, `dst_reg++` for DEST advancement, `#pragma GCC unroll 8` for loop hint.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sign.h`
   **Reason**: Another surviving conditional-branch SFPU kernel showing the `v_if (v < 0.0F)` pattern
   **Key Findings**: Uses the exact `v_if (v < 0.0F)` comparison that leaky_relu would use for its negative-branch guard.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: The generic parameters dispatch function used by all unary SFPU ops including leaky_relu
   **Key Findings**: VectorMode::RC loops 4 faces, calls sfpu_func per face, uses `TTI_SETRWC` between faces to advance DEST pointer.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains `eltwise_unary_sfpu_configure_addrmod` and `_llk_math_eltwise_unary_sfpu_init_`
   **Key Findings**: Standard ADDR_MOD_7 with zero increments for all ops not in the special-case list. Leaky_relu (SfpuType::lrelu) was not in any special case.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Contains the macro infrastructure used by LLK dispatch functions
   **Key Findings**: `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro matches the leaky_relu dispatch pattern (one runtime uint32_t parameter, VectorMode::RC).

10. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
    **Reason**: The dispatch configuration file where LEAKY_RELU's init/func strings were defined
    **Key Findings**: All leaky_relu cases have been removed. The surviving structure shows how parameterized ops are dispatched via `get_op_init_and_func_parameterized()`.

11. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`
    **Reason**: Surviving compute API header showing the exact pattern for tile-level API functions
    **Key Findings**: Pattern: `ALWI void op_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_op<APPROX>(idst))); }` -- leaky_relu followed this pattern with an additional `param0` argument.

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
    **Reason**: The `Converter::as_float()` utility used by parameterized SFPU kernels
    **Key Findings**: Simple union-based uint32_t-to-float reinterpretation, used to decode the bit-cast slope parameter.
