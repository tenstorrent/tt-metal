## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Status: UNIMPLEMENTED (Post-Nuke Stub)

**XIELU has no SFPU kernel implementation in the current codebase.** The operation was removed during the SFPU "nuke" (a codebase-wide removal of SFPU kernel implementations) and has **not yet been reimplemented**. What remains is a skeleton of host-side declarations and disabled bindings.

The following subsections document exactly what exists and what is missing, so that a future implementation can be informed by this analysis.

### Unary Dispatch Summary
- **UnaryOpType**: `XIELU` (enum value 124 in `unary_op_types.hpp`)
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` which falls through to `default: return "eltwise_sfpu.cpp"`)
- **SFPU_OP_CHAIN_0 expansion**: **DOES NOT EXIST** -- calling `get_op_init_and_func_parameterized()` with `UnaryOpType::XIELU` hits `default: TT_THROW("unexpected parameterized op type {}", op_type)` because there is no case for XIELU

#### Why Dispatch Fails

There are two independent failures in the dispatch path:

1. **`is_parametrized_type(UnaryOpType::XIELU)` returns `false`** (`unary_op_utils.hpp` line 44-51). Only `HARDTANH` and `SOFTSHRINK` are listed. Since the C++ `ttnn::xielu()` function passes parameters `{alpha_p, alpha_n}` via `UnaryWithParam`, the `params` span is non-empty, so `get_op_init_and_func()` routes to `get_op_init_and_func_parameterized()`.

2. **`get_op_init_and_func_parameterized()` has no case for XIELU** (`unary_op_utils.cpp` line 41-43). It falls through to `default: TT_THROW(...)`. This means any attempt to create the program will crash at compile-define generation time.

As a result, the full dispatch chain (`prim::unary` -> `UnaryProgramFactory::create` -> `get_block_defines` -> `get_op_init_and_func`) terminates with an exception before any compute kernel is compiled.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(XIELU)` falls through to `default: return false` in `unary_op_utils.cpp` |
| Template parameter (SFPU_OP_CHAIN) | N/A -- no init/func generated | No case for XIELU in `get_op_init_and_func_parameterized()` |
| Effective SFPU path | N/A -- dispatch crashes before reaching SFPU | `TT_THROW` at define generation |

### SFPU Abstraction Layers

No SFPU abstraction layers exist for this operation.

| Layer | File Path |
|-------|-----------|
| **API Header** | Does not exist -- no `xielu_tile()` or `xielu_tile_init()` function in `compute_kernel_api/` |
| **LLK Dispatch** | Does not exist -- no `llk_math_eltwise_unary_sfpu_xielu` function |
| **Core SFPU Implementation** | Does not exist -- no `ckernel_sfpu_xielu.h` file |
| **Parameters Dispatch** | Does not exist |

### Call Chain

No call chain can be traced because the dispatch path terminates with an exception at `get_op_init_and_func_parameterized()` before any kernel code is generated or compiled.

The intended flow would be:
1. `ttnn::xielu(input, alpha_p, alpha_n)` -> `unary_impl()` -> `prim::unary()` -> `UnaryProgramFactory::create()`
2. `create()` calls `utils::get_block_defines(op_chain, "0", "0", dtype)` which calls `get_defines_impl()` -> `get_op_init_and_func(XIELU, {alpha_p, alpha_n}, "0", dtype)`
3. Since `params` is non-empty, routes to `get_op_init_and_func_parameterized(XIELU, {alpha_p, alpha_n}, "0", dtype)`
4. **CRASH**: `default: TT_THROW("unexpected parameterized op type {}", op_type)`

### What Exists (Host-Side Skeleton)

The following components exist but cannot be exercised:

1. **Enum value**: `UnaryOpType::XIELU` in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:124`
2. **C++ API function**: `ttnn::xielu()` in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:169-177`
3. **C++ declaration**: `ttnn::xielu()` in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:235-240` with defaults `alpha_p=0.8f, alpha_n=0.8f`
4. **Disabled nanobind binding**: `bind_xielu()` in `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:966-1012`, inside `#if 0` block (line 913), never called from `py_module()`
5. **Test**: `test_xielu()` in `tests/ttnn/unit_tests/operations/eltwise/test_activation.py:482-497` (would fail at runtime since Python binding is disabled)
6. **Include guard name**: `SFPU_OP_XIELU_INCLUDE` (referenced in documentation only, not in `get_macro_definition()`)

### Mathematical Definition (from nanobind docstring)

xIELU (Expanded Integral of the Exponential Linear Unit) is a custom piecewise trainable activation function derived from "Deriving Activation Functions Using Integration" (arXiv:2411.13010).

With fixed `beta = 0.5` and `eps = -1e-6`:
```
x > 0  :  alpha_p * x^2 + beta * x
x <= 0 :  alpha_n * (expm1(min(x, eps))) - (alpha_n * x) + 0.5 * x
```

Where:
- `alpha_p` (default 0.8): positive slope parameter
- `alpha_n` (default 0.8): negative slope parameter
- `beta` (hardcoded 0.5): linear coefficient
- `eps` (hardcoded -1e-6): small negative constant for numerical stability in `expm1`

**Positive branch**: Quadratic `alpha_p * x^2 + 0.5 * x`
**Negative branch**: `alpha_n * (exp(min(x, eps)) - 1) - alpha_n * x + 0.5 * x` which simplifies to `alpha_n * expm1(min(x, eps)) + (0.5 - alpha_n) * x`

### SFPU Instructions Used

No SFPU instructions are used -- no SFPU kernel implementation exists.

**If implemented**, the operation would likely require:
- `SFPLOAD` / `SFPSTORE`: Move data between DEST and LREGs
- `SFPMAD`: Multiply-add for quadratic computation (`alpha_p * x * x + beta * x`)
- `SFPMUL`: For `alpha_n * expm1(...)` and `alpha_n * x`
- `SFPEXEXP` / `SFPMAD` chain: For `expm1()` approximation (similar to existing `exp` kernel)
- `SFPSETCC` / `SFPENCC`: Condition code for the piecewise branch (`x > 0` vs `x <= 0`)
- `SFPLOADI`: Load immediate constants (`alpha_p`, `alpha_n`, `beta=0.5`, `eps=-1e-6`)
- `SFPMIN` or CC-guarded move: For `min(x, eps)` in the negative branch

### SFPU Register Usage

No registers are used -- no SFPU kernel implementation exists.

### Address Mode Configuration

No address mode is configured -- no SFPU kernel implementation exists.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Identify XIELU enum value
   **Key Findings**: `XIELU` is defined at line 124, between LOGSIGMOID and LGAMMA

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Trace dispatch path for XIELU through `get_op_init_and_func`, `get_op_approx_mode`, `get_compute_kernel_path`
   **Key Findings**: XIELU has no case in any switch statement. `get_op_init_and_func_parameterized` throws for XIELU. `get_op_approx_mode` returns false (default). `get_compute_kernel_path` returns "eltwise_sfpu.cpp" (default).

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type` for XIELU
   **Key Findings**: `is_parametrized_type` does NOT include XIELU -- only HARDTANH and SOFTSHRINK are listed

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
   **Reason**: Understand how `ttnn::xielu()` is implemented
   **Key Findings**: Creates `UnaryWithParam{UnaryOpType::XIELU, {alpha_p, alpha_n}}` and passes to `unary_impl()`

5. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Find default parameter values
   **Key Findings**: `alpha_p = 0.8f`, `alpha_n = 0.8f`

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
   **Reason**: Check Python binding status and mathematical documentation
   **Key Findings**: `bind_xielu()` is inside `#if 0` block (disabled), not called from `py_module()`. Contains full mathematical formula in docstring.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Verify program factory dispatch path
   **Key Findings**: Standard `UnaryProgramFactory::create` uses `get_block_defines()` which would crash for XIELU

8. **File**: `docs/sfpu_operations/key_notes/xielu_key_notes.md`
   **Reason**: Check for additional implementation notes
   **Key Findings**: Minimal content -- only states "Tenstorrent custom activation function (parametrized)" with no formula details

9. **File**: `docs/sfpu_operations/unary_eltwise_sfpu_list.md`
   **Reason**: Confirm include guard naming
   **Key Findings**: Documents `SFPU_OP_XIELU_INCLUDE` as the include guard for XIELU

10. **File**: `tests/ttnn/unit_tests/operations/eltwise/test_activation.py`
    **Reason**: Check test expectations and golden function usage
    **Key Findings**: Test exists at line 482 using `ttnn.get_golden_function(ttnn.xielu)` with various alpha_p/alpha_n combinations

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU architecture and instruction semantics
    **Key Findings**: Used for speculating about potential SFPU instructions needed for a future implementation
