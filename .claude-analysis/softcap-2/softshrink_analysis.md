## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (default from `get_compute_kernel_path()`, but unreachable at runtime)
- **SFPU_OP_CHAIN_0 expansion**: **NONE** -- `get_op_init_and_func_parameterized()` hits `default: TT_THROW("unexpected parameterized op type {}", op_type)` before any SFPU tile function can be resolved

**CRITICAL FINDING: The SFPU kernel for SOFTSHRINK does not exist in this codebase.**

The operation `SOFTSHRINK` is declared at the type-system level:
1. `UnaryOpType::SOFTSHRINK` exists in the enum at `unary_op_types.hpp:113`
2. `is_parametrized_type(SOFTSHRINK)` returns `true` in `unary_op_utils.hpp:47`
3. `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` is registered in `unary.hpp:165`
4. A nanobind binding exists in `unary_nanobind.cpp:1970` with `lambd` parameter (default 0.5)

However, the SFPU dispatch chain is **broken** at the host-side compile-time define generation:
- `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp:41-43` has no `case UnaryOpType::SOFTSHRINK:` -- it falls through to `default: TT_THROW(...)`
- `get_op_init_and_func_default()` in `unary_op_utils.cpp:46-55` also has no `SOFTSHRINK` case
- No `softshrink_tile()` API function exists anywhere in the codebase
- No `softshrink_tile_init()` API function exists anywhere in the codebase
- No `ckernel_sfpu_softshrink.h` file exists in any LLK directory
- No `_calculate_softshrink` function exists anywhere in the codebase
- No `llk_math_eltwise_unary_sfpu_softshrink` function exists anywhere in the codebase

Attempting to run `ttnn.softshrink(tensor, lambd)` at runtime would cause a `TT_THROW` exception during program compilation (when `get_block_defines()` calls `get_op_init_and_func()` for the SOFTSHRINK op type).

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSHRINK)` in `unary_op_utils.cpp:73-77` -- no explicit case, falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | N/A -- dispatch throws before reaching template resolution | `get_op_init_and_func_parameterized()` has no SOFTSHRINK case |
| Effective SFPU path | **Unreachable** -- `TT_THROW` before any SFPU code is invoked | `unary_op_utils.cpp:42` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist -- no `softshrink_tile()` function |
| **LLK Dispatch** | This level of abstraction doesn't exist -- no `llk_math_eltwise_unary_sfpu_softshrink` |
| **Core SFPU Implementation** | This level of abstraction doesn't exist -- no `ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | This level of abstraction doesn't exist |

### Call Chain
No call chain exists. The dispatch path terminates at the host-side define generation:

1. `ttnn::softshrink(input, lambd)` calls `ttnn::detail::unary_impl()` with `UnaryOpType::SOFTSHRINK` and `param0=lambd`
2. `UnaryProgramFactory::create()` calls `utils::get_block_defines(op_chain, ...)` (at `unary_program_factory.cpp:115`)
3. `get_block_defines()` calls `get_defines_impl()` which calls `get_op_init_and_func(SOFTSHRINK, params=[lambd], ...)`
4. Since `params` is non-empty, `get_op_init_and_func_parameterized()` is called
5. The switch statement has no `case UnaryOpType::SOFTSHRINK:` -- falls to `default: TT_THROW("unexpected parameterized op type {}", op_type)`
6. **Exception thrown. No compute kernel is ever compiled. No SFPU code runs.**

### Parameters Dispatch Summary
N/A -- no parameters dispatch exists for this operation.

### Annotated SFPU Kernel Source
No SFPU kernel source exists for SOFTSHRINK. The following files were searched and confirmed absent:
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_sfpu/ckernel_sfpu_softshrink.h` -- does not exist
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_sfpu/ckernel_sfpu_softshrink.h` -- does not exist
- `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h` -- does not exist
- Any file containing `_calculate_softshrink` or `softshrink_tile` -- does not exist

### SFPU Instructions Used
N/A -- no SFPU kernel implementation exists.

### SFPU Register Usage
N/A -- no SFPU kernel implementation exists.

### Address Mode Configuration
N/A -- no SFPU kernel implementation exists.

### Mathematical Definition (for reference)
The softshrink function is defined as (from PyTorch documentation and `docs/sfpu_operations/wave2_instructions.md`):

```
softshrink(x, lambda) =
    x - lambda    if x > lambda
    x + lambda    if x < -lambda
    0             otherwise
```

Where `lambda` (default = 0.5) is a non-negative threshold parameter.

The backward operation exists in `unary_backward.cpp:725-738` and implements:
```
softshrink_bw(grad, input, lambd) = where(input < -lambd OR input > lambd, grad, 0.0)
```

### Context: Why This Operation Lacks an SFPU Kernel

Based on `docs/sfpu_operations/wave2_instructions.md`, SOFTSHRINK was planned as a "Wave 2" operation for the `vignjatijevic-sfpu-generator` code generation pipeline. The document explicitly notes:

> "SOFTSHRINK has `UnaryOpType::SOFTSHRINK` in the enum but **NO** `REGISTER_UNARY_OPERATION` macro in `unary.hpp` and **NO** nanobind binding."

Since that documentation was written, the registration macro and nanobind binding have been added (they exist in the current code), but the underlying SFPU compute infrastructure was never completed:
- No case was added to `get_op_init_and_func_parameterized()` for `SOFTSHRINK`
- No `softshrink_tile()` / `softshrink_tile_init()` API functions were created
- No LLK dispatch function was created
- No `ckernel_sfpu_softshrink.h` kernel was created

The operation is a **type-system stub** with host-side registration but no device-side SFPU implementation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Primary dispatch file for SFPU operations -- needed to determine `get_op_init_and_func` and `get_compute_kernel_path` for SOFTSHRINK
   **Key Findings**: No case for `UnaryOpType::SOFTSHRINK` in either `get_op_init_and_func_parameterized` (falls to `TT_THROW`) or `get_op_init_and_func_default` (also `TT_THROW`). `get_op_approx_mode` returns `false` (default case). `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default case).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Contains `is_parametrized_type` function
   **Key Findings**: `SOFTSHRINK` returns `true` from `is_parametrized_type`, meaning it expects a float parameter (lambda)

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Enum definition for unary op types
   **Key Findings**: `SOFTSHRINK` exists at line 113 in the `UnaryOpType` enum

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: TTNN-level operation registration macros
   **Key Findings**: `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)` at line 165 provides the C++ API entry point

5. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
   **Reason**: Python binding registration
   **Key Findings**: Nanobind binding exists at line 1970 with `lambd` parameter (default 0.5)

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Program factory that would compile and dispatch the compute kernel
   **Key Findings**: No special handling for SOFTSHRINK (unlike HARDSHRINK which gets a tmp CB). The `get_block_defines` call at line 115 would throw for SOFTSHRINK.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp`
   **Reason**: Backward implementation provides the mathematical definition
   **Key Findings**: `softshrink_bw` at line 725 implements the backward using `ttnn::where`, `ttnn::lt`, `ttnn::gt`, confirming the piecewise mathematical definition

8. **File**: `docs/sfpu_operations/wave2_instructions.md`
   **Reason**: Planning document for SFPU operation generation waves
   **Key Findings**: SOFTSHRINK was planned for Wave 2 generation. The document explicitly notes the registration gap and lists the mathematical definition: `x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise`

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware reference for instruction semantics and register model
   **Key Findings**: Used as background reference for understanding SFPU architecture, though no SFPU instructions are used by this operation since no kernel exists

10. **File**: `.claude/references/diagram-templates.md`
    **Reason**: Template for CC state machine diagrams
    **Key Findings**: Not applicable -- no CC manipulation to diagram since no kernel exists
