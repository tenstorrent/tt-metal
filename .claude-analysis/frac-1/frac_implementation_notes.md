# frac — Implementation Notes

## Overview
The `frac` operation computes the fractional part of a floating-point number: `frac(x) = x - floor(x)`.

The SFPU kernel layer was already fully implemented in the LLK (Low-Level Kernel) submodule. The implementation only required wiring up the integration layers in tt-metal/ttnn.

## Implementation Approach

The LLK already provides:
- `_calculate_frac_()` in `ckernel_sfpu_rounding_ops.h` — uses `_trunc_body_()` then computes `x - trunc(x)`
- `frac_tile(uint32_t idst)` in `rounding.h` — compute API wrapper
- `rounding_op_tile_init()` — shared init function for all rounding ops

The `UnaryOpType::FRAC` enum entry and `REGISTER_UNARY_OPERATION(frac, FRAC)` already existed but the dispatch was not connected.

## Changes Made

### 1. Old-style dispatch registration (unary_op_utils.cpp)
Added `FRAC` (and `FLOOR`, `CEIL`, `TRUNC`) to `get_op_init_and_func_default()`:
- init: `"rounding_op_tile_init();"`
- func: `fmt::format("frac_tile({});", idst)`

### 2. Python nanobind binding (unary_nanobind.cpp)
Added `bind_unary_operation<"frac", &ttnn::frac>(...)` with LaTeX doc string.
Also added bindings for `floor`, `ceil`, `trunc` as they share the same pattern.

### 3. Golden function (unary.py)
Added `frac` to `TTNN_ELTWISE_UNARY_CPP_FUNCTIONS` list and `name_to_golden_function` dict.
Maps to `torch.frac`. Also added `floor`, `ceil`, `trunc` golden functions.

### 4. Test file
Created `tests/ttnn/unit_tests/operations/eltwise/test_frac.py` with:
- Parameterized shape and dtype tests
- Negative input tests
- Integer input tests (frac should return 0)
- Special value tests
- Large value tests

## Reference Operations Used
- **trunc** — frac is defined as `x - trunc(x)` in the SFPU kernel
- **floor/ceil** — same rounding family, share init function
- **softsign/silu** — pattern for old-style dispatch and nanobind registration

## Design Decisions
1. Used `x - trunc(x)` (not `x - floor(x)`) matching the LLK implementation. Note: PyTorch `torch.frac` also uses truncation semantics (not floor), so `frac(-2.7) = -0.7`.
2. No custom include macro needed — rounding ops are part of standard LLK includes.
3. Added floor/ceil/trunc alongside frac since they were equally missing from old dispatch.

## Known Limitations
- Precision for very large values is limited by bfloat16 representation
- The SFPU kernel uses truncation (not floor) so `frac(x) = x - trunc(x)`, matching PyTorch semantics

### New Files
tests/ttnn/unit_tests/operations/eltwise/test_frac.py

### Modified Files
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
