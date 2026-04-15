## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**STATUS: NO SFPU KERNEL IMPLEMENTATION EXISTS**

The `tanhshrink` operation (`tanhshrink(x) = x - tanh(x)`) is registered as `UnaryOpType::TANHSHRINK` in the unary operation type system but does **not** have a functioning SFPU kernel implementation in this version of the codebase. Attempting to invoke it through the device operation path would result in a runtime error (`TT_THROW("unexpected op type")`).

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: N/A -- no SFPU kernel exists
- **SFPU_OP_CHAIN_0 expansion**: N/A -- `get_op_init_and_func_default()` has no case for `TANHSHRINK`; falls through to `default: TT_THROW("unexpected op type")`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` in `unary_op_utils.cpp` -- returns `false` (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | N/A | No init/func dispatch exists for TANHSHRINK |
| Effective SFPU path | N/A -- operation would throw before reaching any SFPU code | `get_op_init_and_func_default()` line 53: `default: TT_THROW("unexpected op type")` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist |
| **LLK Dispatch** | This level of abstraction doesn't exist |
| **Core SFPU Implementation** | This level of abstraction doesn't exist |
| **Parameters Dispatch** | This level of abstraction doesn't exist |

### Call Chain

No call chain exists. The dispatch path terminates with an exception:

1. `ttnn::tanhshrink(input)` (from `REGISTER_UNARY_OPERATION` macro in `unary.hpp:158`)
2. `ttnn::detail::unary_impl(input, {UnaryWithParam{UnaryOpType::TANHSHRINK}}, ...)` (inline function created by macro)
3. `ttnn::prim::unary(...)` (in `unary.cpp:45`)
4. `UnaryDeviceOperation::launch(...)` (in `unary_device_operation.cpp:213`)
5. `UnaryProgramFactory::create(...)` which calls `utils::get_block_defines(op_chain, ...)` on line 115
6. `get_block_defines()` -> `get_defines_impl()` -> `get_op_init_and_func()` -> `get_op_init_and_func_default()`
7. **TERMINATES**: `default: TT_THROW("unexpected op type {}", op_type)` at `unary_op_utils.cpp:53`

### Evidence of Non-Implementation

The following evidence confirms the absence of an SFPU kernel for `tanhshrink`:

1. **No dispatch case**: `get_op_init_and_func_default()` in `unary_op_utils.cpp` (lines 46-55) handles only `FRAC`, `SWISH`, `ATANH`, and `SINH`. `TANHSHRINK` falls through to `default: TT_THROW`.

2. **No SFPU kernel file**: No `ckernel_sfpu_tanhshrink.h` exists anywhere in `tt_metal/third_party/tt_llk/` or `tt_metal/hw/ckernels/`.

3. **No compute kernel API**: No `tanhshrink_tile()` or `tanhshrink_tile_init()` function exists in the compute kernel API headers.

4. **No nanobind forward binding**: The `unary_nanobind.cpp` file does not bind `tanhshrink` for the forward path (only the backward `tanhshrink_bw` is bound in `unary_backward_nanobind.cpp`).

5. **Commented-out composite**: `unary_composite_op.cpp:501` contains only a comment `// tanhshrink(x) = x - tanh(x)` with no active implementation.

### Historical Context

The `tanhshrink` operation was previously implemented as a **composite operation** (host-side decomposition into `x - tanh(x)`) rather than a dedicated SFPU kernel. The comment at `unary_composite_op.cpp:501` is a remnant of this prior implementation. The operation appears to have been "nuked" (removed) as part of a larger refactoring effort, and a dedicated SFPU kernel was never created to replace it.

The backward operation (`tanhshrink_bw`) still functions correctly because it composes existing operations at the host level: `torch.square(torch.tanh(input)) * grad_data`, implemented as `ttnn::multiply(grad, ttnn::square(ttnn::tanh(input)))`.

### Parameters Dispatch Summary

N/A -- no parameters dispatch exists for this operation.

### Annotated SFPU Kernel Source

N/A -- no SFPU kernel source exists for this operation.

### SFPU Instructions Used

N/A -- no SFPU instructions are used because no SFPU kernel exists.

### SFPU Register Usage

N/A -- no SFPU registers are used because no SFPU kernel exists.

### Address Mode Configuration

N/A -- no address mode is configured because no SFPU kernel exists.

### Mathematical Definition

For reference, `tanhshrink` is defined as:

```
tanhshrink(x) = x - tanh(x)
```

A potential SFPU implementation would need to:
1. Load `x` from DEST
2. Compute `tanh(x)` (using `SFPNONLINEAR` with InstrMod=5 for hardware-accelerated tanh, or the existing `_calculate_tanh_()` function from `ckernel_sfpu_tanh.h`)
3. Subtract `tanh(x)` from `x` (using `SFPMAD` with sign inversion: `x * 1.0 + (-tanh(x))`)
4. Store the result back to DEST

## Local Knowledge Sources
### Local References
1. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
   **Reason**: Verify UnaryOpType::TANHSHRINK enum exists
   **Key Findings**: TANHSHRINK is defined at line 111

2. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"
   **Reason**: Trace SFPU_OP_CHAIN_0 dispatch path for TANHSHRINK
   **Key Findings**: get_op_init_and_func_default() has no case for TANHSHRINK -- falls through to default: TT_THROW. get_op_approx_mode() returns false for all ops. get_compute_kernel_path() returns "eltwise_sfpu.cpp" for all ops.

3. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
   **Reason**: Check is_parametrized_type and function declarations
   **Key Findings**: TANHSHRINK is not listed as a parametrized type

4. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp"
   **Reason**: Trace REGISTER_UNARY_OPERATION macro expansion
   **Key Findings**: REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK) at line 158 creates a C++ inline function that calls unary_impl with UnaryOpType::TANHSHRINK and no params

5. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp"
   **Reason**: Trace unary_impl dispatch to prim::unary
   **Key Findings**: unary_impl calls prim::unary which launches UnaryDeviceOperation

6. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp"
   **Reason**: Trace device operation to program factory
   **Key Findings**: select_program_factory returns UnaryProgramFactory for non-sharded inputs

7. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp"
   **Reason**: Trace program factory dispatch of SFPU_OP_CHAIN_0 defines
   **Key Findings**: Calls utils::get_block_defines(args.op_chain, "0", "0", input.dtype()) which leads to the TT_THROW

8. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp"
   **Reason**: Check for composite implementation of tanhshrink
   **Key Findings**: Only a comment at line 501 -- no active implementation

9. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp"
   **Reason**: Check if tanhshrink has a Python binding
   **Key Findings**: No binding for tanhshrink forward op

10. **File**: "ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp"
    **Reason**: Check backward implementation for context
    **Key Findings**: tanhshrink_bw (lines 941-947) uses host-level composition: ttnn::square(ttnn::tanh(input)) * grad

11. **File**: ".claude/references/sfpu-hardware-model.md"
    **Reason**: Reference for SFPU architecture and SFPNONLINEAR instruction (InstrMod=5 for tanh)
    **Key Findings**: SFPNONLINEAR with InstrMod=5 provides hardware-accelerated tanh approximation
