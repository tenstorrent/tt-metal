## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**NOTE**: The LEAKY_RELU operation has been fully removed ("nuked") from this repository clone as part of the deep nuke evaluation environment (Phase 2, commit `8c0af4489d`). All kernel source code, dispatch cases, enum values, and registrations have been deleted. This analysis is reconstructed from surviving artifacts: the nuke manifest (`DEEP_NUKE_MANIFEST.md`), API documentation (`docs/source/.../relu_tile.rst`), the breadcrumb logging reference (`sfpu-operation-analyzer.md`), the operation catalog (`unary_eltwise_sfpu_list_with_links.md`), and the surviving infrastructure patterns of other unary SFPU operations.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU` (removed from enum in Phase 2 nuke)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (generic, no specialized kernel)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, param0)` where `param0` is the `negative_slope` float bit-cast to `uint32_t`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- switch has only `default: return false` case (LEAKY_RELU had no explicit case before removal) |
| Template parameter (SFPU_OP_CHAIN) | `param0` (negative_slope as uint32) | `get_op_init_and_func_parameterized()` -- LEAKY_RELU was a parameterized type with `negative_slope` passed as the first parameter |
| Effective SFPU path | The `APPROX` template parameter defaults to `false`. The SFPU kernel receives `negative_slope` as a runtime parameter, not as a compile-time approximation switch. The kernel is a simple piecewise-linear operation that does not have distinct approximate/exact code paths. | The `_calculate_lrelu_` function's `APPROXIMATION_MODE` template parameter is not used to select between different algorithm paths for this operation |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` (shared with RELU family -- **DELETED** in nuke) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_relu.h` (**DELETED** in nuke) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_relu.h` (shared with RELU family -- **EMPTIED** to `#pragma once` only) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic, still exists) |

**Quasar variant**: Had a separate file `ckernel_sfpu_lrelu.h` (entirely **DELETED** in nuke).

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `leaky_relu_tile(0, param0)`, calling the tile-level API.
2. **API Header** (`relu.h`): `leaky_relu_tile(uint32_t idst, uint32_t slope)` calls `MATH((llk_math_eltwise_unary_sfpu_lrelu<APPROX>(idst, slope)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_relu.h`): `llk_math_eltwise_unary_sfpu_lrelu<APPROX>()` calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_lrelu_<APPROX, 8>, dst_index, (int)VectorMode::RC, slope)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_<APPROX>()` sets up DEST addressing, stalls for SFPU availability, then calls the SFPU function 4 times (once per face in RC mode), with `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_relu.h`): `_calculate_lrelu_<APPROX, 8>(slope)` executes the leaky ReLU computation using raw TTI instructions over 8 iterations per face.

The init path is analogous:
1. `leaky_relu_tile_init()` calls `llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>()`.
2. This calls `llk_math_eltwise_unary_sfpu_init<SfpuType::lrelu, APPROX>()`.
3. Which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::lrelu>()`, configuring SFPU registers and address modes.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces of the tile (full 32x32 = 1024 elements).
- **Operation invocation**: The core SFPU function `_calculate_lrelu_<APPROX, ITERATIONS>` is called once per face with `ITERATIONS=8`. Each call processes one 16x16 face (256 elements) in 8 sfpi iterations of 32 elements each.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_face_addr` between faces). Within a face, `dst_reg++` advances 1 sfpi row = 2 physical DEST rows = 32 elements. Between faces, Wormhole uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice (= +16 physical rows = 1 face), while Blackhole uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

**IMPORTANT**: The SFPU kernel source code has been **deleted** from this repository clone as part of the deep nuke evaluation. The files `ckernel_sfpu_relu.h` (WH/BH) have been emptied to just `#pragma once`, and `ckernel_sfpu_lrelu.h` (quasar) has been entirely deleted. The function `_calculate_lrelu_` no longer exists in the codebase.

Based on the evidence gathered from the nuke manifest, logging reference examples, API documentation, and the mathematical definition of leaky ReLU (`max(0, x) + negative_slope * min(0, x)`, equivalently `x if x >= 0, else negative_slope * x`), the original kernel used raw TTI instructions (Style B) with a simple LT0 condition code guard pattern.

The kernel's algorithm was:
1. Load the `negative_slope` parameter into an LREG (via `SFPLOADI` or passed as function argument)
2. For each of 8 iterations per face:
   a. Load the current element from DEST into an LREG (`SFPLOAD`)
   b. Test if the value is negative using condition codes (`SFPSETCC` with LT0 mode)
   c. For negative elements only: multiply by `negative_slope` (`SFPMUL` or `SFPMAD`)
   d. Store the result back to DEST (`SFPSTORE`)
   e. Advance to next sfpi row

The following is a **reconstructed approximation** of the original kernel, based on the instruction list from the logging reference (`SFPLOADI`, `SFPLOAD`, `SFPSETCC`, `SFPMUL`, `SFPENCC`, `SFPSTORE`) and the CC pattern (`simple_LT0_guard`):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
// [RECONSTRUCTED -- original source deleted in deep nuke Phase 2]
// This function was part of the RELU family shared header.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_lrelu_(const uint32_t slope) {
    // slope is negative_slope bit-cast from float to uint32_t
    // Load negative_slope into an LREG for use in multiplication
    // The exact LREG assignment and SFPLOADI encoding are not recoverable.

    for (int d = 0; d < ITERATIONS; d++) {
        // Load current element from DEST
        // SFPLOAD dst_reg[d] -> LREG

        // Test if value < 0 using SFPSETCC (CC_LT0 mode)
        // SFPENCC to enable conditional execution

        // For negative lanes only: multiply by negative_slope
        // SFPMUL value * slope -> result

        // SFPENCC to disable conditional execution (all lanes active)

        // Store result back to DEST
        // SFPSTORE LREG -> dst_reg[d]

        // Advance dst_reg (handled by loop / dst_reg++)
    }
}
```

**Why raw TTI rather than SFPI?** The relu family historically used raw TTI instructions for performance optimization. A simple piecewise-linear activation like leaky ReLU benefits from direct hardware instruction control: a single `SFPSETCC` + conditional `SFPMUL` is more efficient than the v_if/v_else SFPI abstraction overhead. The logging reference consistently classifies this kernel as `B_raw_TTI` with a `simple_LT0_guard` CC pattern.

### SFPU Instructions Used

| Instruction | Description | Role in Leaky ReLU |
|------------|-------------|-------------------|
| `SFPLOADI` | Load 16-bit immediate to LREG | Load the `negative_slope` constant into an LREG (done once before the loop, or encoded as immediate) |
| `SFPLOAD` | Load from DEST row to LREG with format conversion | Load current tile element from DEST register into LREG for processing |
| `SFPSETCC` | Set CC.Res based on comparison (LT0 mode) | Test whether the loaded value is negative; sets per-lane CC.Res = 1 for lanes where value < 0 |
| `SFPMUL` | Multiply with immediate or register | Multiply negative elements by `negative_slope`; only executes on lanes where CC.Res = 1 (value < 0) |
| `SFPENCC` | Enable/disable CC, set/clear CC.Res | Enable CC masking before the SFPSETCC test, and disable it after the conditional multiply to restore all-lane execution |
| `SFPSTORE` | Store LREG to DEST row with format conversion | Write the processed element (either unchanged or scaled by negative_slope) back to DEST |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST** | Source and destination for tile elements. Each sfpi address accesses 32 elements (2 physical rows x 16 elements). 8 iterations per face, 4 faces per tile. |
| **LREG (slope)** | Holds the `negative_slope` parameter loaded via `SFPLOADI`. This register is loaded once and reused across all iterations. |
| **LREG (value)** | Holds the current tile element loaded from DEST via `SFPLOAD`. Used for the sign test and conditional multiplication. |
| **CC (Condition Code)** | Per-lane condition code used to mask the multiplication. `SFPSETCC` in LT0 mode sets CC.Res = 1 for negative lanes. `SFPENCC` enables/disables the CC mechanism. Only negative elements are modified; positive elements pass through unchanged. |

### Address Mode Configuration

The address mode for LEAKY_RELU (SfpuType `lrelu`) follows the **default** configuration in `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()`. Since `lrelu` is not in any of the special-cased `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `signbit`, etc.), only `ADDR_MOD_7` is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `ADDR_MOD_7.srca.incr` | 0 | No SrcA increment |
| `ADDR_MOD_7.srcb.incr` | 0 | No SrcB increment |
| `ADDR_MOD_7.dest.incr` | 0 | No DEST auto-increment (DEST progression is managed by `dst_reg++` in the SFPU kernel and `SETRWC`/`inc_dst_addr` between faces) |

This configuration is **identical on both Wormhole B0 and Blackhole**. Both architectures set the same `ADDR_MOD_7` with all-zero increments for the `lrelu` SfpuType. The face-to-face DEST progression is handled at the params dispatch level, not by the address mode.

## Local Knowledge Sources
### Local References
1. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Authoritative record of what was removed in the deep nuke and how
   **Key Findings**: LEAKY_RELU removed in Phase 2 (commit `8c0af4489d`). Enum value, registration, dispatch all deleted. `_calculate_lrelu_` removed from `ckernel_sfpu_relu.h` (WH/BH). `ckernel_sfpu_lrelu.h` (quasar) deleted entirely. SfpuType `lrelu` removed from `llk_sfpu_types.h`.

2. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/relu_tile.rst`
   **Reason**: API documentation for the compute kernel tile-level functions
   **Key Findings**: `leaky_relu_tile_init()` and `leaky_relu_tile(uint32_t idst, uint32_t lower_limit)` -- confirms the API signature with two parameters (idst + slope as uint32).

3. **File**: `docs/sfpu_operations/key_notes/leaky_relu_key_notes.md`
   **Reason**: Mathematical formula and parameter documentation
   **Key Findings**: Formula is `max(0, x) + negative_slope * min(0, x)`, parameter `negative_slope` with default 0.01.

4. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list_with_links.md`
   **Reason**: Operation catalog with parameterization info
   **Key Findings**: LEAKY_RELU is parameterized with `negative_slope` param, maps to `torch.nn.LeakyReLU`.

5. **File**: `tt_metal/third_party/tt_ops_code_gen/references/logging/sfpu-operation-analyzer.md`
   **Reason**: Breadcrumb logging reference contains example analysis events for leaky_relu
   **Key Findings**: Kernel style `B_raw_TTI`, instructions `SFPLOADI/SFPLOAD/SFPSETCC/SFPMUL/SFPENCC/SFPSTORE`, CC pattern `simple_LT0_guard`, init function `leaky_relu_tile_init()`, tile function `leaky_relu_tile(idst, param0)`, include guard `SFPU_OP_RELU_FAMILY_INCLUDE`.

6. **File**: `tt_metal/third_party/tt_ops_code_gen/skills/sfpu-unary-nuke-op/SKILL.md`
   **Reason**: Documents the nuke process and RELU family structure
   **Key Findings**: LEAKY_RELU was part of `SFPU_OP_RELU_FAMILY_INCLUDE` family (with RELU, RELU6, RELU_MAX, RELU_MIN). Shared `relu.h` API header and `ckernel_sfpu_relu.h` ckernel.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Dispatch infrastructure -- confirms LEAKY_RELU removal from all switch statements
   **Key Findings**: `get_op_approx_mode()` has only `default: return false`. `get_op_init_and_func_parameterized()` has only `default: TT_THROW`. `get_compute_kernel_path()` has only `default: return "eltwise_sfpu.cpp"`. All LEAKY_RELU cases removed.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Parameterized type check
   **Key Findings**: `is_parametrized_type()` no longer lists LEAKY_RELU (only HARDTANH and SOFTSHRINK remain).

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and init infrastructure
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType>()` sets `ADDR_MOD_7` with all-zero increments as default. `lrelu` SfpuType was not in any special-case branch.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Parameters dispatch -- VectorMode::RC face iteration pattern
    **Key Findings**: RC mode iterates 4 faces, calling the SFPU function once per face. Face advancement via `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x2 on Wormhole.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Blackhole parameters dispatch for comparison
    **Key Findings**: Same RC mode logic, uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of raw `TTI_SETRWC`.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU instruction semantics, register model, addressing model
    **Key Findings**: SFPSETCC modes (LT0 = mode 0), SFPENCC modes, SFPMUL semantics, stride-2 addressing model, ITERATIONS=8 per face derivation.
