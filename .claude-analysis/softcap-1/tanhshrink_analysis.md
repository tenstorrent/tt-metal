## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**CRITICAL FINDING: `tanhshrink` has NO dedicated SFPU kernel implementation in this codebase.** The operation is registered as `UnaryOpType::TANHSHRINK` and declared via the `REGISTER_UNARY_OPERATION` macro, but the SFPU dispatch path is incomplete. Calling `ttnn::tanhshrink()` at runtime would result in a `TT_THROW("unexpected op type {}")` because there is no case for `TANHSHRINK` in `get_op_init_and_func_default()`. No `tanhshrink_tile`, `tanhshrink_tile_init`, or `ckernel_sfpu_tanhshrink.h` file exists anywhere in the codebase.

The mathematical definition `tanhshrink(x) = x - tanh(x)` (confirmed by a comment in `unary_composite_op.cpp:501`) indicates this was historically implemented as a composite operation decomposed into `tanh` + `subtract`. The dedicated SFPU kernel was never implemented.

### Unary Dispatch Summary
- **UnaryOpType**: `TANHSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (default from `get_compute_kernel_path()`, but dispatch would fail before the kernel is reached)
- **SFPU_OP_CHAIN_0 expansion**: **NONE** -- `get_op_init_and_func_default()` has no case for `TANHSHRINK` and throws `TT_THROW("unexpected op type {}")` at the `default` branch

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANHSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | No `get_op_init_and_func` case exists for TANHSHRINK -- dispatch throws before reaching any template |
| Effective SFPU path | **N/A -- operation throws at dispatch** | `get_op_init_and_func_default()` line 53: `default: TT_THROW("unexpected op type {}", op_type)` |

### SFPU Abstraction Layers
No SFPU abstraction layers exist for this operation because the SFPU kernel was never implemented.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist |
| **LLK Dispatch** | This level of abstraction doesn't exist |
| **Core SFPU Implementation** | This level of abstraction doesn't exist |
| **Parameters Dispatch** | This level of abstraction doesn't exist |

### Call Chain
The call chain terminates at the host-side dispatch:

1. `ttnn::tanhshrink(input)` -- inline function from `REGISTER_UNARY_OPERATION` macro in `unary.hpp:158`
2. `ttnn::detail::unary_impl(input, {UnaryWithParam{UnaryOpType::TANHSHRINK}}, ...)` -- in `unary.cpp:22`
3. `ttnn::prim::unary(...)` -- in `unary_device_operation.hpp:43`
4. `UnaryProgramFactory::create(...)` -- would call `utils::get_block_defines()`
5. `get_block_defines()` -> `get_defines_impl()` -> `get_op_init_and_func()` -> `get_op_init_and_func_default()` -- **THROWS** at `default: TT_THROW("unexpected op type {}", op_type)` in `unary_op_utils.cpp:53`

The operation never reaches device-side execution.

### Parameters Dispatch Summary
No parameters dispatch exists for this operation. The dispatch chain fails at the host-side `get_op_init_and_func_default()` function.

### Annotated SFPU Kernel Source
No SFPU kernel source exists for `tanhshrink`. There is no `ckernel_sfpu_tanhshrink.h` file and no `_calculate_tanhshrink` function anywhere in the codebase.

The mathematical definition `tanhshrink(x) = x - tanh(x)` suggests that if implemented as a dedicated SFPU kernel, it would:
1. Load the input value from DEST
2. Compute `tanh(x)` (potentially reusing the existing `tanh` SFPU kernel logic or the `SFPNONLINEAR` instruction with InstrMod=5)
3. Subtract the `tanh(x)` result from the original `x`
4. Store the result back to DEST

### SFPU Instructions Used
No SFPU instructions are used because the kernel does not exist. If implemented, the operation would likely use:
- `SFPLOAD` -- to load input from DEST
- `SFPNONLINEAR` (InstrMod=5) -- hardware-accelerated `tanh` approximation, OR a software `tanh` via `SFPMAD` chains
- `SFPMAD` -- for the subtraction `x - tanh(x)` (since `vFloat + vFloat` maps to `SFPMAD`)
- `SFPSTORE` -- to store the result back to DEST

### SFPU Register Usage
No registers are used because the kernel does not exist.

### Address Mode Configuration
No address mode configuration exists because the kernel does not exist.

### Evidence of Stub Status

The following evidence confirms that `TANHSHRINK` is a stub operation with no SFPU kernel:

1. **Enum registration**: `UnaryOpType::TANHSHRINK` exists in `unary_op_types.hpp:111`
2. **Header-only declaration**: `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` in `unary.hpp:158` creates an inline C++ function
3. **No dispatch case**: `get_op_init_and_func_default()` in `unary_op_utils.cpp:46-55` has no case for `TANHSHRINK` -- the `default` branch throws `TT_THROW`
4. **No include guard**: `get_macro_definition()` in `unary_op_utils.cpp:18-26` has no case for `TANHSHRINK` -- falls through to the default `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`
5. **No SFPU kernel file**: No `ckernel_sfpu_tanhshrink.h` exists in `tt_metal/third_party/tt_llk/`
6. **No tile-level API**: No `tanhshrink_tile` or `tanhshrink_tile_init` function exists in any header
7. **No Python binding**: The forward operation is not bound in `unary_nanobind.cpp` (only the backward `tanhshrink_bw` is bound)
8. **Composite origin**: Comment in `unary_composite_op.cpp:501` reads `// tanhshrink(x) = x - tanh(x)`, confirming the operation was historically a composite decomposition

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Primary file for understanding SFPU dispatch -- `get_op_init_and_func_default()`, `get_op_approx_mode()`, `get_compute_kernel_path()`
   **Key Findings**: `TANHSHRINK` has no case in `get_op_init_and_func_default()` (would TT_THROW), no case in `get_op_approx_mode()` (falls to default: false), and no case in `get_compute_kernel_path()` (falls to default: eltwise_sfpu.cpp)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Enum definition for all unary op types
   **Key Findings**: `TANHSHRINK` is listed at line 111 as a valid enum value

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Macro registration and inline function declaration for tanhshrink
   **Key Findings**: `REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)` at line 158 creates the inline function that calls `detail::unary_impl`

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
   **Reason**: Implementation of `detail::unary_impl` which dispatches to the device operation
   **Key Findings**: `detail::unary_impl` calls `prim::unary()` which goes to the program factory

5. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp`
   **Reason**: Historical context for the tanhshrink operation
   **Key Findings**: Comment at line 501 reads `// tanhshrink(x) = x - tanh(x)`, confirming composite decomposition origin

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
   **Reason**: Checked whether tanhshrink has Python bindings
   **Key Findings**: No forward `tanhshrink` binding exists; only `tanhshrink_bw` is bound in the backward file

7. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware reference for understanding what instructions would be needed
   **Key Findings**: SFPNONLINEAR InstrMod=5 provides hardware-accelerated tanh; SFPMAD handles addition/subtraction
