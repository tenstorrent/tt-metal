## SFPU Kernel Implementation

**STATUS: OPERATION NOT FOUND IN CURRENT CODEBASE**

The `rrelu` (Randomized Leaky ReLU) operation does not exist as an implemented SFPU kernel in the current codebase. This analysis documents the absence and provides context for what the operation would implement.

### Evidence of Absence

| Check | Result |
|-------|--------|
| `UnaryOpType::RRELU` in `unary_op_types.hpp` | **Not found** — enum does not contain RRELU |
| `ckernel_sfpu_rrelu.h` | **Not found** — no glob matches for `**/ckernel_sfpu_rrelu*` |
| `llk_math_eltwise_unary_sfpu_rrelu.h` | **Not found** — no glob matches for `**/llk*rrelu*` |
| `api/compute/eltwise_unary/rrelu.h` | **Not found** — no glob matches for `**/compute/**/rrelu*` |
| Any `rrelu` match in `unary_op_utils.cpp` | **Not found** — grep returns zero matches |
| Any `rrelu` match in `tt_metal/` source tree | **Not found** — grep returns zero matches |
| `docs/sfpu_operations/key_notes/rrelu_key_notes.md` | **Exists** — contains formula and "Note: NEW - to be implemented." |

Git history shows the operation was previously implemented (commit `24376c2fcb9`) but subsequently removed in a batch deletion. Per analysis rules, deleted code cannot be retrieved from git history.

### Unary Dispatch Summary

- **UnaryOpType**: `RRELU` — **does not exist** in the current `UnaryOpType` enum
- **Compute kernel**: N/A (operation not registered)
- **SFPU_OP_CHAIN_0 expansion**: N/A (operation not registered)

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | N/A | Operation not in `get_op_approx_mode()` switch |
| Template parameter (SFPU_OP_CHAIN) | N/A | Operation not in `get_op_init_and_func()` switch |
| Effective SFPU path | N/A | No SFPU implementation exists |

### Mathematical Definition

From `docs/sfpu_operations/key_notes/rrelu_key_notes.md`:

```
f(x) = x           if x >= 0
f(x) = a * x       if x < 0
```

Where:
- **Training mode**: `a` is sampled from `Uniform(lower, upper)` per forward pass
- **Evaluation mode**: `a = (lower + upper) / 2` (fixed)
- **Default parameters**: `lower = 1/8 (0.125)`, `upper = 1/3 (0.333...)`
- **PyTorch reference**: `torch.nn.RReLU`

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | Does not exist |
| **LLK Dispatch** | Does not exist |
| **Core SFPU Implementation** | Does not exist |
| **Parameters Dispatch** | Does not exist |

### Call Chain

No call chain can be traced — the operation has no source files in the current codebase.

### Parameters Dispatch Summary

N/A — operation not implemented.

### Annotated SFPU Kernel Source

No kernel source exists in the current codebase. The operation's SFPU kernel (`ckernel_sfpu_rrelu.h`) was removed.

### SFPU Instructions Used

N/A — no kernel to analyze.

### SFPU Register Usage

N/A — no kernel to analyze.

### Address Mode Configuration

N/A — no kernel to analyze.

### Structural Notes for Future Implementation

Based on the mathematical definition and related operations that DO exist in the codebase (e.g., `PRELU_SFPU`, `LEAKY_RELU`, `CELU`, `ELU`, `SELU`), the following structural observations apply:

1. **Formula similarity**: RRELU in evaluation mode is identical to Leaky ReLU / PReLU with slope `(lower + upper) / 2`. The SFPU kernel would use the same conditional-multiply pattern: `v_if(x < 0) { x = x * slope; } v_endif`.

2. **Parameter passing**: The operation requires two parameters (`lower`, `upper`). In the previous implementation, these were passed as `uint32_t` runtime arguments via `Converter::as_float()` — the same pattern used by `CELU` (two params: `alpha`, `alpha_recip`) and `SELU` (two params: `scale`, `alpha`).

3. **Training vs. evaluation**: The SFPU implementation likely had two modes — a training mode using on-device random number generation (similar to `DROPOUT`) and an evaluation mode using simple conditional multiply (similar to `PRELU_SFPU`).

4. **Custom LLK header pattern**: The previous implementation used a custom LLK header (`llk_math_eltwise_unary_sfpu_rrelu.h`) with a two-parameter `_llk_math_eltwise_unary_sfpu_params_` dispatch, rather than the common LLK library.

5. **SFPU_OP_RRELU_INCLUDE**: The include guard pattern would follow the `SFPU_OP_*_INCLUDE` convention used by other custom-LLK operations.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Checked for `RRELU` in the `UnaryOpType` enum
   **Key Findings**: RRELU is not present in the enum (lines 20-135)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked for rrelu dispatch, init/func strings, and approx mode
   **Key Findings**: No rrelu references. Only IDENTITY, DROPOUT, MISH, and LOGIT have explicit handling.

3. **File**: `docs/sfpu_operations/key_notes/rrelu_key_notes.md`
   **Reason**: Only remaining rrelu artifact in the codebase
   **Key Findings**: Contains mathematical formula (`x if x >= 0, a*x if x < 0`), parameters (lower/upper), training/eval modes, and "Note: NEW - to be implemented."

4. **File**: `.claude-analysis/selu-1/reference_selection.md`
   **Reason**: Referenced rrelu as a structural reference for selu implementation
   **Key Findings**: Described rrelu as having `ckernel_sfpu_rrelu.h`, `llk_math_eltwise_unary_sfpu_rrelu.h`, two-param dispatch pattern, and `Converter::as_float()` usage. These files no longer exist.

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Checked for SFPU_OP_RRELU_INCLUDE define
   **Key Findings**: File is empty (contains only `#pragma once`). All SFPU split includes were removed.
