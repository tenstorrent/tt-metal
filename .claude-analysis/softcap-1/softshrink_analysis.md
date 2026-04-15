## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**CRITICAL FINDING: The SFPU kernel for softshrink has been intentionally removed ("nuked") from this branch.** This branch (`2026_04_15_0838_run1_softcap_sfpu_gen_1`) is a controlled evaluation environment where SFPU unary operation implementations were surgically deleted (documented in `DEEP_NUKE_MANIFEST.md`, Phase 1). The softshrink SFPU kernel was removed across all abstraction layers:

- **Compute API header** (`softshrink.h`): Deleted from `tt_metal/hw/inc/api/compute/eltwise_unary/` (confirmed in `sources.cmake` build fix, Phase 1)
- **LLK dispatch** (`llk_math_eltwise_unary_sfpu_softshrink.h`): Deleted from `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/`
- **Core SFPU kernel** (`ckernel_sfpu_softshrink.h`): Deleted from `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/`
- **Host dispatch** (`get_op_init_and_func_parameterized`): SOFTSHRINK case removed from `unary_op_utils.cpp`
- **SfpuType enum**: No `softshrink` entry exists in `llk_sfpu_types.h`

What survives:
- `UnaryOpType::SOFTSHRINK` in the enum (`unary_op_types.hpp:113`)
- `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` in `unary.hpp:165`
- `is_parametrized_type()` returning `true` for SOFTSHRINK (`unary_op_utils.hpp:47`)
- Python nanobind binding (`unary_nanobind.cpp:1970`)
- Tests (`test_activation.py:388`, sweep tests)

Calling `ttnn.softshrink()` at runtime would throw a `TT_THROW("unexpected parameterized op type {}")` in `get_op_init_and_func_parameterized` because no SOFTSHRINK case exists.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (would-be; determined by `get_compute_kernel_path()` default case)
- **SFPU_OP_CHAIN_0 expansion**: **MISSING** -- no `softshrink_tile(0)` or equivalent exists. The dispatch chain is broken at `get_op_init_and_func_parameterized()`.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- SOFTSHRINK has no explicit case; falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | **N/A -- dispatch broken** | `get_op_init_and_func_parameterized()` has no SOFTSHRINK case; would throw before any template argument is resolved |
| Effective SFPU path | **No SFPU path reachable** | The operation cannot be dispatched to any SFPU code in the current state of this branch |

### SFPU Abstraction Layers

All layers for softshrink have been deleted. The table below documents the expected paths based on the standard unary SFPU pattern and the naming convention observed in analogous operations (frac, hardtanh, swish).

| Layer | File Path |
|-------|-----------|
| **API Header** | DELETED -- would be `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` |
| **LLK Dispatch** | DELETED -- would be `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h` |
| **Core SFPU Implementation** | DELETED -- would be `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared, still exists) |

### Call Chain

The call chain for softshrink does not exist in this codebase. Based on the standard unary SFPU pattern observed in analogous operations (e.g., frac), the expected call chain would be:

1. `softshrink_tile(idst)` (API header) calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst)` (LLK dispatch)
2. `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_softshrink_<APPROX, 8>, dst_index, VectorMode::RC, param0)` -- passing the lambda parameter as a runtime argument
3. `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces in RC mode, calling the core function once per face with 8 SFPU iterations, using `TTI_SETRWC` to advance between faces

### Parameters Dispatch Summary

The parameters dispatch function `_llk_math_eltwise_unary_sfpu_params_` (from `llk_math_eltwise_unary_sfpu_params.h`) is shared infrastructure that still exists. For softshrink, the expected configuration would be:

- **Vector mode**: `VectorMode::RC` -- all 4 faces processed (standard for element-wise unary operations)
- **Operation invocation**: The core SFPU function would be called once per face (4 times total for RC mode). Each call processes 8 SFPU iterations (one per face, ITERATIONS=8 default).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces). On Wormhole, the init function sets `ADDR_MOD_7` with all increments = 0 (the default for most SFPU ops, as softshrink does not require special address modes like typecast or topk).

### Annotated SFPU Kernel Source

**No SFPU kernel source exists for softshrink on this branch.** The kernel was deleted as part of the Deep Nuke Phase 1 operation.

#### Mathematical Definition

Softshrink is a piecewise linear activation function:
```
softshrink(x, lambda) =
    x - lambda    if x > lambda
    x + lambda    if x < -lambda
    0             otherwise (i.e., -lambda <= x <= lambda)
```

#### Expected Implementation Pattern (based on analogous hardtanh kernel)

The closest surviving reference is `ckernel_sfpu_hardtanh.h`, which implements a piecewise linear function using SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`). Softshrink would follow the same "Piecewise Linear" family pattern (as classified in the DEEP_NUKE_MANIFEST).

The hardtanh kernel uses pre-negated parameters and additive offset tricks to implement threshold comparisons with minimal SFPU instructions. A softshrink kernel would similarly use SFPI conditional constructs (`v_if`, `v_endif`) to implement the three-branch piecewise logic, loading the lambda parameter via `sfpi::s2vFloat16b()` from a uint32 parameter.

The expected kernel structure (based on the hardtanh pattern) would be:

```cpp
// EXPECTED pattern (NOT actual code -- softshrink kernel was deleted)
// File would be: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_softshrink.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_softshrink_(const int iterations, std::uint32_t param0) {
    // param0 = lambda in FP16_B format
    sfpi::vFloat lambda = sfpi::s2vFloat16b(param0);

    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        v_if (val > lambda) {
            result = val - lambda;
        } v_elseif (val < -lambda) {
            result = val + lambda;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

**Note**: This is a hypothetical reconstruction, NOT the actual deleted implementation. The actual implementation may have used different optimization strategies (e.g., the additive offset trick used in hardtanh).

### SFPU Instructions Used

No SFPU instructions can be documented because the kernel does not exist. Based on the expected SFPI-based implementation pattern:

| Instruction | Expected Usage |
|-------------|---------------|
| `SFPLOAD` | Load element from DEST register into LREG (via `dst_reg[0]`) |
| `SFPSTORE` | Store result back to DEST register (via `dst_reg[0] = result`) |
| `SFPMAD` | Float addition (`val - lambda`, `val + lambda`) -- emitted as `a * 1.0 + b` |
| `SFPSETCC` | Conditional comparison for `v_if (val > lambda)` and `v_if (val < -lambda)` |
| `SFPENCC` | Enable/disable condition code for `v_if`/`v_endif` blocks |
| `SFPCOMPC` | Complement CC for `v_elseif`/`v_else` branches |
| `SFPPUSHC` | Push CC state for nested conditionals (if `v_elseif` is used) |
| `SFPPOPC` | Pop CC state after nested conditionals |
| `SFPLOADI` | Load immediate for `0.0f` (the default result in the dead zone) and for `-lambda` |

### SFPU Register Usage

No register usage can be documented because the kernel does not exist. Based on the expected pattern:

| Register | Expected Usage |
|----------|---------------|
| `dst_reg[0]` (LREG mapped to DEST) | Input element read and result write-back |
| LREG (for `lambda`) | Holds the lambda parameter converted from FP16_B via `s2vFloat16b` |
| LREG (for `-lambda`) | May hold negated lambda for the `x < -lambda` comparison |
| LREG (for `result`) | Holds the intermediate computation result (0.0, val-lambda, or val+lambda) |
| LREG (for `0.0f`) | Constant zero for the dead zone output |

### Address Mode Configuration

The SFPU init function `eltwise_unary_sfpu_configure_addrmod` (from `llk_math_eltwise_unary_sfpu.h`) sets `ADDR_MOD_7` for most SFPU operations:

**Wormhole B0** (from `llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

Softshrink would use this default address mode since it is a standard element-wise operation with no special address mode requirements. The DEST address progression within each face is handled by `dst_reg++` in the SFPU loop (stride-2, advancing 2 physical DEST rows per iteration), and the inter-face advancement is handled by `TTI_SETRWC` in `_llk_math_eltwise_unary_sfpu_params_`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, dispatch configuration, and approximation mode for SOFTSHRINK
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_parameterized()` has NO SOFTSHRINK case -- would throw at runtime.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check if SOFTSHRINK is a parametrized type and its registration
   **Key Findings**: `is_parametrized_type(UnaryOpType::SOFTSHRINK)` returns `true`. SOFTSHRINK expects a float parameter (lambda).

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Verify SOFTSHRINK exists in the UnaryOpType enum
   **Key Findings**: `SOFTSHRINK` is at line 113 in the enum, between THRESHOLD and HARDSHRINK.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Verify the registration macro used for softshrink
   **Key Findings**: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` at line 165. Takes `float parameter` as the lambda value.

5. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Understand why the SFPU kernel is missing
   **Key Findings**: Softshrink was deleted in Phase 1 "Wave 4 Deep Nuke" (commit `efdc0ad853`). Classified as "Piecewise Linear" family. Dispatch removed, compute API deleted, metal ckernel deleted, metal LLK deleted, tests deleted. Enum values and registration macros intentionally kept.

6. **File**: `docs/sfpu_operations/wave2_instructions.md`
   **Reason**: Historical context on softshrink as a target operation
   **Key Findings**: Softshrink was a Wave 2 target operation with formula `x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise`. Parametrized with lambda (default=0.5).

7. **File**: `docs/sfpu_operations/key_notes/softshrink_key_notes.md`
   **Reason**: Mathematical definition and parameter details
   **Key Findings**: Formula confirmed as piecewise. Lambda default=0.5, common range [0.1, 1.0]. Deterministic, mode-independent.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Reference for the "Piecewise Linear" family pattern used by softshrink's family
   **Key Findings**: hardtanh uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Takes 3 pre-negated uint32_t params converted via s2vFloat16b. Implements piecewise clamping via additive offset + comparison. Loop processes `iterations` sfpi rows with `dst_reg++` per iteration.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the shared parameters dispatch function
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode::RC by iterating over 4 faces, calling the SFPU function once per face, with TTI_SETRWC between faces. This is the dispatch layer that softshrink would plug into.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Understand the SFPU init function and address mode configuration
    **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` sets ADDR_MOD_7 with all increments=0 as default for most SFPU ops. Softshrink has no special address mode requirements.

11. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
    **Reason**: Reference for the LLK dispatch pattern used by generated Wave 2 ops
    **Key Findings**: Shows the standard pattern: init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>()`, tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` with the calculate function as a callable.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model for instruction semantics and register details
    **Key Findings**: SFPMAD used for float addition (no dedicated add instruction). SFPSETCC for comparisons. SFPENCC/SFPCOMPC/SFPPUSHC/SFPPOPC for conditional execution. SFPLOAD/SFPSTORE for DEST access. Standard per-face ITERATIONS=8.

13. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
    **Reason**: Verify the compute kernel dispatch pattern
    **Key Findings**: Standard SFPU_OP_CHAIN_0 dispatch within tile_regs_acquire/commit/wait/release cycle. This is the kernel softshrink would use via get_compute_kernel_path() default case.

14. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: Check if softshrink has special program factory handling (like HARDSHRINK has)
    **Key Findings**: HARDSHRINK has special CB c_1 allocation and runtime arg packing. SOFTSHRINK has NO special handling in the program factory -- it would use the standard path.
