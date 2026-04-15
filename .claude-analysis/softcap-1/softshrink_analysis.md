## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**CRITICAL FINDING: The softshrink SFPU kernel has been completely deleted from this codebase.** All SFPU implementation layers (compute API header, LLK dispatch, ckernel SFPU function) were removed during the Phase 1 "deep nuke" operation (commit `efdc0ad853`). Only residual host-side infrastructure (enum value, registration macro, parametrized type flag) remains. Attempting to invoke `ttnn::softshrink()` at runtime would hit `TT_THROW("unexpected parameterized op type {}", op_type)` in `get_op_init_and_func_parameterized()`.

### Mathematical Definition

softshrink(x, lambda) is a piecewise linear activation function:
- x - lambda, if x > lambda
- x + lambda, if x < -lambda
- 0, otherwise

Default lambda = 0.5. Common range: [0.1, 1.0].

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` which has only `default: return "eltwise_sfpu.cpp"`)
- **SFPU_OP_CHAIN_0 expansion**: **DELETED** -- `get_op_init_and_func_parameterized()` has no case for `SOFTSHRINK` (throws `TT_THROW` at runtime)

#### What Remains in the Codebase

| Component | Status | Location |
|-----------|--------|----------|
| `UnaryOpType::SOFTSHRINK` enum value | Present | `unary_op_types.hpp:113` |
| `is_parametrized_type(SOFTSHRINK)` returns `true` | Present | `unary_op_utils.hpp:47` |
| `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` | Present | `unary.hpp:165` |
| Python nanobind binding | Present | `unary_nanobind.cpp:1970` |
| `get_op_init_and_func_parameterized()` case for SOFTSHRINK | **DELETED** | `unary_op_utils.cpp:41-43` (switch falls through to `default: TT_THROW`) |
| `get_macro_definition()` case for SOFTSHRINK | **DELETED** | `unary_op_utils.cpp:18-26` (switch falls through to `default: SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`) |
| Compute API header `softshrink.h` | **DELETED** | Was at `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` (confirmed deleted by `sources.cmake` build fix, commit `9b2679bc62`) |
| `sfpu_split_includes.h` `#if SFPU_OP_ACTIVATIONS_INCLUDE` guard | **DELETED** | No longer present in `sfpu_split_includes.h` |
| `activations.h` include of softshrink | **DELETED** | `activations.h` is now just `#pragma once` |
| `ckernel_sfpu_softshrink.h` (WH) | **DELETED** | Was at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` |
| `ckernel_sfpu_softshrink.h` (BH) | **DELETED** | Was at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/` |
| `llk_math_eltwise_unary_sfpu_softshrink.h` (WH) | **DELETED** | Was at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` |
| `llk_math_eltwise_unary_sfpu_softshrink.h` (BH) | **DELETED** | Was at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/` |

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- only has `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | N/A -- dispatch deleted | `get_op_init_and_func_parameterized()` has no case for SOFTSHRINK |
| Effective SFPU path | N/A -- no SFPU kernel exists | All SFPU implementation files have been deleted |

### SFPU Abstraction Layers

All SFPU layers have been deleted. The table below documents what existed before the nuke based on the standard unary SFPU pattern and the `DEEP_NUKE_MANIFEST.md`.

| Layer | File Path |
|-------|-----------|
| **API Header** | **DELETED** -- was `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` |
| **LLK Dispatch** | **DELETED** -- was `llk_math_eltwise_unary_sfpu_softshrink.h` (WH and BH) |
| **Core SFPU Implementation** | **DELETED** -- was `ckernel_sfpu_softshrink.h` (WH and BH), or possibly inside `ckernel_sfpu_activations.h` (which is now stubbed) |
| **Parameters Dispatch** | Uses the shared `llk_math_eltwise_unary_sfpu_params.h` (still exists at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

The call chain for softshrink **cannot be traced** because the SFPU kernel files have been deleted. Based on the standard unary SFPU pattern observed in surviving operations (e.g., swish):

1. `SFPU_OP_CHAIN_0` would expand to `softshrink_tile_init(); softshrink_tile(0);` (or similar)
2. `softshrink_tile(idst)` in `softshrink.h` would call `MATH((llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst)))`
3. `llk_math_eltwise_unary_sfpu_softshrink()` in `llk_math_eltwise_unary_sfpu_softshrink.h` would call `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0)`
4. The params dispatch would iterate over 4 faces, calling the core SFPU function 4 times with SETRWC between faces
5. `calculate_softshrink()` in `ckernel_sfpu_softshrink.h` would implement the piecewise logic using SFPI abstractions

### Parameters Dispatch Summary

Since softshrink is a parametrized operation (lambda as a float parameter), its dispatch would have:

- **Vector mode**: `VectorMode::RC` (all 4 faces processed) -- this is the standard default for unary operations
- **Operation invocation**: The shared `_llk_math_eltwise_unary_sfpu_params_` function would call the core SFPU function once per face (4 times total), with `TTI_SETRWC` advancing the DEST address between faces
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces)
- **Parameter passing**: The lambda value would be passed as a `uint32_t` runtime argument (bit-cast from float) through the `_llk_math_eltwise_unary_sfpu_params_` variadic `Args&&... args` mechanism

### Annotated SFPU Kernel Source

**No source code exists.** The SFPU kernel has been deleted from the codebase.

Based on the mathematical definition and the pattern observed in surviving SFPU kernels (e.g., `ckernel_sfpu_swish.h`), a softshrink SFPU kernel would need to implement the following piecewise logic using SFPI abstractions:

```
For each of ITERATIONS (8) sfpi rows:
  1. Load x from dst_reg[0]
  2. If x > lambda:  result = x - lambda
  3. Else if x < -lambda:  result = x + lambda
  4. Else:  result = 0
  5. Store result to dst_reg[0]
  6. Advance dst_reg++
```

This would require:
- `sfpi::vFloat` for x and lambda
- `v_if` / `v_elseif` / `v_else` / `v_endif` for the piecewise conditions
- Standard `dst_reg[0]` load/store pattern
- The lambda parameter would need to be reconstructed from a `uint32_t` bit-pattern

### SFPU Instructions Used

**No SFPU instructions can be listed** because the kernel implementation does not exist.

For a hypothetical softshrink implementation using SFPI abstractions, the compiler would likely emit:
- **SFPLOAD** -- load x from DEST register
- **SFPMAD** -- arithmetic operations (x - lambda, x + lambda, realized as fused multiply-add)
- **SFPLOADI** -- load the lambda constant and zero constant
- **SFPSETCC** -- set condition codes for the piecewise comparisons (x > lambda, x < -lambda)
- **SFPENCC** -- enable/disable condition code masking for v_if/v_else blocks
- **SFPCOMPC** -- complement condition code for else branches
- **SFPPUSHC** / **SFPPOPC** -- CC stack management for nested conditionals
- **SFPSTORE** -- store result back to DEST register

### SFPU Register Usage

**No register usage can be documented** because the kernel implementation does not exist.

For a hypothetical implementation:
- **DEST registers**: Input tile data (read via `dst_reg[0]`), output written back to same location
- **LREGs**: Would hold intermediate values (x, lambda, comparison results, arithmetic results)
- **Constant registers**: Lambda value would likely be loaded once and reused across iterations

### Address Mode Configuration

**No address mode configuration can be documented** because the kernel implementation does not exist.

The standard unary SFPU address mode configuration (used by surviving operations like swish) is:
- **Wormhole B0**: `ADDR_MOD_2` -- standard SFPU addressing with stride-2 auto-increment
- **Blackhole**: `ADDR_MOD_6` (or equivalent) -- same logical behavior, different register encoding

### Nuke Context

softshrink was part of the **SFPU_OP_ACTIVATIONS_INCLUDE** operation family (along with SOFTSIGN, HARDSIGMOID, and CELU). The entire family's SFPU implementations were deleted during the Phase 1 deep nuke. The `activations.h` aggregation header was emptied to just `#pragma once`, and the `sfpu_split_includes.h` guard for `SFPU_OP_ACTIVATIONS_INCLUDE` was removed.

The softshrink operation is categorized as "Piecewise Linear" in the nuke manifest, alongside hardsigmoid, hardtanh, and hardswish. This classification reflects that softshrink's mathematical definition is a piecewise linear function with three segments separated by the threshold lambda.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Primary dispatch file for unary operations -- checked for SOFTSHRINK case in get_op_init_and_func_parameterized(), get_op_approx_mode(), get_compute_kernel_path(), get_macro_definition()
   **Key Findings**: No SOFTSHRINK case exists in any dispatch function. get_op_init_and_func_parameterized() would throw TT_THROW. get_op_approx_mode() returns false (default). get_compute_kernel_path() returns "eltwise_sfpu.cpp" (default).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Checked is_parametrized_type() and function declarations
   **Key Findings**: SOFTSHRINK returns true from is_parametrized_type(). This confirms it was a parameterized unary operation taking a float lambda parameter.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Checked C++ API registration for softshrink
   **Key Findings**: Line 165: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` -- the registration macro still exists as a stub.

4. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Authoritative record of what was deleted and why
   **Key Findings**: softshrink was deleted in Phase 1 (commit efdc0ad853). Categorized as "Piecewise Linear" family. All layers deleted: dispatch, compute API, ckernel (wh+bh), LLK (wh+bh), tests. Part of SFPU_OP_ACTIVATIONS_INCLUDE family.

5. **File**: `nuke_op_comparison.md`
   **Reason**: Documents the 12 abstraction layers and operation families
   **Key Findings**: SFPU_OP_ACTIVATIONS_INCLUDE family = SOFTSHRINK, SOFTSIGN, HARDSIGMOID, CELU. Confirms the surgical deletion pattern across shared files.

6. **File**: `docs/sfpu_operations/key_notes/softshrink_key_notes.md`
   **Reason**: Mathematical definition reference
   **Key Findings**: Formula: x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise. Default lambda=0.5, common range [0.1, 1.0]. Deterministic, mode-independent.

7. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Checked if SFPU_OP_ACTIVATIONS_INCLUDE guard still exists
   **Key Findings**: The guard has been removed. Only Wave 3 generated ops remain (FRAC, SWISH, ATANH, SINH).

8. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
   **Reason**: Checked if any activation function headers are still included
   **Key Findings**: File is empty (just `#pragma once`). All activation function includes were deleted.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Reference implementation of a surviving SFPU kernel (same generation pattern as softshrink would use)
   **Key Findings**: Shows the SFPI abstraction pattern: template with APPROXIMATION_MODE and ITERATIONS, for loop over ITERATIONS, dst_reg load/store, v_if conditional blocks. This is the pattern softshrink would follow if reimplemented.

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
    **Reason**: Reference LLK dispatch layer for a surviving SFPU kernel
    **Key Findings**: Shows the standard LLK dispatch pattern: init function calls llk_math_eltwise_unary_sfpu_init<SfpuType, APPROXIMATE>(), compute function calls _llk_math_eltwise_unary_sfpu_params_() with the core calculate function as a callable.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Shared parameters dispatch layer (still exists, used by all unary SFPU operations)
    **Key Findings**: Implements VectorMode::RC (4 face loop), VectorMode::R (2 face loop), VectorMode::C (2 face loop with larger SETRWC stride). Uses TTI_SETRWC between faces, TTI_STALLWAIT for SFPU synchronization.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware reference for tile geometry, addressing, instruction semantics
    **Key Findings**: Confirmed stride-2 addressing model, 8 ITERATIONS per face, 32 elements per iteration, SFPMAD for float addition.

13. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
    **Reason**: Compute kernel that dispatches SFPU work
    **Key Findings**: Standard dispatch pattern with SFPU_OP_CHAIN_0 macro expansion. This kernel is shared by all unary SFPU operations.
