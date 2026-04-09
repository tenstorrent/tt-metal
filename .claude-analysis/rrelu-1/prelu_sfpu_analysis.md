## SFPU Kernel Implementation

**NOTE**: This analysis documents the `PRELU_SFPU` operation whose implementation has been **deep-nuked** from this repository clone (Phase 2 of the deep nuke, commit `8c0af4489d`). All source files for the SFPU kernel, LLK dispatch, compute API header, and unary_op_utils dispatch cases have been deleted. This analysis is **reconstructed** from:
- The `DEEP_NUKE_MANIFEST.md` which documents exactly what was removed
- Documentation files (`docs/sfpu_operations/`, `docs/source/tt-metalium/`)
- Surviving structurally-similar operations (swish, hardtanh, threshold, dropout)
- The surviving LLK infrastructure (macros, params dispatch, init)
- Previous analysis breadcrumb logs from the reference discoverer

Items marked `[NUKED]` denote files or code that no longer exist in this repo clone and are reconstructed from references.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile(0, param0)` (where `param0` is the `weight` parameter packed as `uint32_t` via `bit_cast<uint32_t>(weight)`)

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` `[NUKED]` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | `get_op_init_and_func()` -- `PRELU_SFPU` case used `prelu_tile_init()` / `prelu_tile({idst}, {param0})` with no explicit template argument; the API header's `prelu_tile_init()` calls `llk_math_eltwise_unary_sfpu_prelu_init<APPROX>()` where `APPROX` resolves to the compute kernel's `APPROX` define, which defaults to `false` when `math_approx_mode=false` `[NUKED]` |
| Effective SFPU path | `APPROXIMATION_MODE=false` -- the `_calculate_lrelu_` function was called with `APPROXIMATION_MODE=false`. The PRELU_SFPU kernel logic does not branch on this parameter since simple conditional multiply needs no approximation. `[NUKED]` |

### SFPU Abstraction Layers

All files in this table have been deleted from the deep-nuked repo. Paths are reconstructed from documentation and surviving patterns.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` `[NUKED]` -- exposed `prelu_tile(uint32_t idst, uint32_t param0)` and `prelu_tile_init()` (documented in `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/prelu_tile.rst`) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_prelu.h` `[NUKED]` -- contained `llk_math_eltwise_unary_sfpu_prelu<APPROXIMATE>()` and `llk_math_eltwise_unary_sfpu_prelu_init<APPROXIMATE>()` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_relu.h` `[NUKED - emptied to #pragma once]` -- originally contained `_calculate_lrelu_()` which was shared by both LEAKY_RELU and PRELU_SFPU (per `DEEP_NUKE_MANIFEST.md`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (surviving) -- the generic `_llk_math_eltwise_unary_sfpu_params_` template that dispatches the SFPU function with `VectorMode::RC` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `prelu_tile(0, param0)`.
2. **API Header** (`prelu.h` `[NUKED]`): `prelu_tile(idst, param0)` calls `MATH((llk_math_eltwise_unary_sfpu_prelu<APPROX>(idst, param0)))` on the math thread.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_prelu.h` `[NUKED]`): Uses `SFPU_UNARY_ONE_PARAM_KERNEL_FN(_calculate_lrelu_, RC, APPROXIMATE, dst_index, param0)` macro which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_lrelu_<APPROXIMATE>, dst_index, (int)VectorMode::RC, param0)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces in `VectorMode::RC` mode, calling `_calculate_lrelu_<false>(param0)` 4 times (once per face, 8 SFPU iterations per face).
5. **Core SFPU** (`ckernel_sfpu_relu.h` `[NUKED]`): `_calculate_lrelu_<false>(param0)` performs the element-wise PReLU computation: `x >= 0 ? x : weight * x`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (standard for element-wise unary operations).
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` template function calls the SFPU functor (`_calculate_lrelu_<APPROXIMATE>`) once per face in a `for (face = 0; face < 4; face++)` loop. Each invocation processes 8 SFPU iterations (one face = 256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is set with `{srca.incr=0, srcb.incr=0, dest.incr=0}` (address increments handled by `SETRWC` between faces). On Blackhole, the same `ADDR_MOD_7` configuration is used. Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows, 32 elements) per iteration. Between faces, `SETRWC(CR_D, 8)` is called twice (advancing by 16 physical rows = 1 face).

### Annotated SFPU Kernel Source

The original `_calculate_lrelu_` function has been deleted from `ckernel_sfpu_relu.h` (gutted to `#pragma once` in Phase 2/3 of the deep nuke). The following is a **reconstruction** based on:
1. The `DEEP_NUKE_MANIFEST.md` confirming `_calculate_lrelu_` was the function name
2. The `prelu_tile.rst` documenting `prelu_tile(uint32_t idst, uint32_t param0)` signature
3. Surviving structurally-similar kernels (threshold, hardtanh) that use the same `v_if`/`v_endif` SFPI pattern with parameter conversion via `s2vFloat16b()`
4. The mathematical definition: `max(0, x) + weight * min(0, x)` which simplifies to `x >= 0 ? x : weight * x`

The PRELU_SFPU and LEAKY_RELU operations shared the same `_calculate_lrelu_` kernel function. PRELU_SFPU passed the `weight` parameter (from `torch.nn.PReLU`) as `param0`, while LEAKY_RELU passed `negative_slope`. Both are semantically identical at the SFPU level.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
// [RECONSTRUCTED - original file deep-nuked]

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_lrelu_(const uint32_t slope) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Convert the packed FP16_B slope parameter to a vFloat
    sfpi::vFloat s = sfpi::s2vFloat16b(slope);       // weight (for PReLU) or negative_slope (for leaky_relu)

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];            // SFPLOAD: load current element from DEST

        v_if(v < 0.0f) {                              // SFPSETCC: test sign bit (CC.Res = v < 0)
            v *= s;                                    // SFPMAD: multiply negative values by slope
        }
        v_endif;                                       // SFPENCC: disable CC masking

        sfpi::dst_reg[0] = v;                          // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;                               // advance to next sfpi row
    }
}
```

**Wormhole and Blackhole implementations were identical** -- both used SFPI abstractions with the same logic. The Quasar variant was in a separate file `ckernel_sfpu_lrelu.h` (also deleted).

### SFPU Instructions Used

The following instructions are emitted by the SFPI compiler from the reconstructed kernel above. These are the standard instruction mappings for the SFPI abstractions used.

| Instruction | SFPU Opcode | Emitted By | Purpose |
|-------------|-------------|------------|---------|
| `SFPLOAD` | `0x70` | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into LREG0 |
| `SFPLOADI` | `0x71` | `s2vFloat16b(slope)` | Load the 16-bit packed slope constant into an LREG (done once before the loop, or hoisted by compiler) |
| `SFPSETCC` | `0x7B` | `v < 0.0f` (inside `v_if`) | Set CC.Res based on sign bit test -- CC.Res = 1 where element < 0 |
| `SFPMAD` | `0x84` | `v *= s` | Fused multiply-add: `v = v * s + 0.0` (multiply negative elements by slope) |
| `SFPENCC` | `0x8A` | `v_endif` | Disable CC masking, restoring all lanes to active |
| `SFPSTORE` | `0x72` | `dst_reg[0] = v` (write) | Store result back to DEST row pair |

**Note on `v_if(v < 0.0f)`**: The SFPI compiler emits this as:
1. `SFPENCC` with mode to enable CC masking (CC.En = 1, CC.Res = 1)
2. `SFPSETCC` with `SFPSETCC_MOD1_LREG_LT0` to test the sign bit

The `SFPMAD` inside the `v_if` block is CC-guarded -- it only executes on lanes where the condition passed (element < 0). For non-negative elements, the original value is preserved because the guarded multiply does not execute.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register -- holds the loaded element value `v` from DEST, and the result of `v * s` for negative elements |
| **LREG1** (or compiler-assigned) | Holds the slope parameter `s` converted from the packed `uint32_t` via `s2vFloat16b()`. This is loaded once before the iteration loop and reused across all 8 iterations. |
| **DEST rows** | Source and destination for element data. Accessed via stride-2 addressing through `dst_reg[0]` / `dst_reg++`. Each access covers 2 physical rows = 32 elements. |
| **CC register** | Per-lane condition code used for the `v_if(v < 0.0f)` / `v_endif` block. CC.Res is set to 1 for negative elements (sign bit = 1), masking the multiply to only those lanes. |

### Address Mode Configuration

The PRELU_SFPU operation used `SfpuType::prelu` (or similar, now removed from the enum) for the `eltwise_unary_sfpu_configure_addrmod` call during initialization. Since `SfpuType::prelu` did not match any of the special-cased `if constexpr` branches (which are for `topk_local_sort`, `typecast`, `unary_max/min`, `signbit`, and on Blackhole also `reciprocal`), it fell through to the default configuration:

**Both Wormhole and Blackhole**:
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is the standard configuration for SFPU unary operations. The DEST address increment is 0 because SFPU address progression is handled internally by the SFPU engine (via `dst_reg++` which maps to hardware register counter updates), not by the address mode auto-increment mechanism.

No additional address modes (ADDR_MOD_6, etc.) are configured for this operation -- only the standard ADDR_MOD_7 is used.

## Local Knowledge Sources
### Local References
1. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: To understand exactly what was removed and what the original structure looked like
   **Key Findings**: PRELU_SFPU was removed in Phase 2 (commit `8c0af4489d`). It shared `_calculate_lrelu_` in `ckernel_sfpu_relu.h` with LEAKY_RELU. The enum value, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER, and binary composite `prelu()` were all removed. SfpuType was `lrelu` (shared with LEAKY_RELU).

2. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list.md`
   **Reason**: To determine the include guard and parametrization of PRELU_SFPU
   **Key Findings**: PRELU_SFPU used `SFPU_OP_PRELU_INCLUDE` as its include guard. It was parametrized with a `weight` parameter.

3. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list_with_links.md`
   **Reason**: To confirm PyTorch equivalent and parameter details
   **Key Findings**: PRELU_SFPU maps to `torch.nn.PReLU` with a `weight` parameter.

4. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/prelu_tile.rst`
   **Reason**: To confirm the compute API signature
   **Key Findings**: `prelu_tile_init()` and `prelu_tile(uint32_t idst, uint32_t param0)` were the API functions.

5. **File**: `docs/sfpu_operations/key_notes/prelu_sfpu_key_notes.md`
   **Reason**: To understand the mathematical formula
   **Key Findings**: Formula is `max(0, x) + weight * min(0, x)`, equivalent to `x >= 0 ? x : weight * x`.

6. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: To understand how parametrized SFPU operations are dispatched via macros
   **Key Findings**: `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro dispatches a one-parameter SFPU function through `_llk_math_eltwise_unary_sfpu_params_`. This is the macro that PRELU_SFPU's LLK dispatch would have used.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: To understand the face-iteration dispatch pattern
   **Key Findings**: VectorMode::RC iterates over 4 faces, calling the SFPU functor once per face. Between faces, `SETRWC(CR_D, 8)` is called twice (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) to advance by one face stride.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: To understand the address mode configuration for SFPU init
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` sets `ADDR_MOD_7` with `{dest.incr=0}` for all non-special-cased operations. PRELU_SFPU falls through to the default.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: As a surviving reference for the SFPI `v_if` conditional pattern with parameter conversion
   **Key Findings**: Threshold uses `sfpi::vFloat v_threshold = Converter::as_float(threshold)` and `v_if(in <= v_threshold)` -- structurally similar to PRELU's `v_if(v < 0.0f) { v *= s; }` pattern.

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
    **Reason**: As a surviving fully-implemented SFPI-style kernel for comparison
    **Key Findings**: Swish uses the same SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`) and `#pragma GCC unroll` pattern. Confirmed the SFPI style used by operations in the relu family.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
    **Reason**: As a surviving parametrized SFPU kernel showing `s2vFloat16b()` parameter conversion
    **Key Findings**: Hardtanh receives uint32_t parameters and converts them with `sfpi::s2vFloat16b(param)`. This is the same pattern PRELU_SFPU used for its weight parameter.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU instruction semantics and register model
    **Key Findings**: SFPSETCC with `SFPSETCC_MOD1_LREG_LT0` tests the sign bit. SFPMAD is used for float multiply (v * s + 0.0). SFPENCC disables CC masking.

13. **File**: `.claude-analysis/rrelu-1/reference_selection.md`
    **Reason**: To understand the relationship between PRELU_SFPU and the target operation (rrelu)
    **Key Findings**: PRELU_SFPU has the same formula as leaky_relu. The slope parameter is a runtime value passed from C++ dispatch. This pattern directly maps to rrelu's eval-mode negative-branch multiply.
