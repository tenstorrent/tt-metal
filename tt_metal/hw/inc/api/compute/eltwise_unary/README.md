# How to Add a New Eltwise Unary Operator Using Macros

This document describes the recommended approach for adding a new eltwise unary operator in `tt-metal` using the macro system introduced in `llk_math_eltwise_unary_sfpu_macros.h`. This replaces the previous pattern of intermediate function wrappers and per-op header files.

## Why Macros?

Previously, each unary op (e.g., eqz, log1p, max, negative) required a dedicated intermediate header (e.g., `llk_math_eltwise_unary_sfpu_eqz.h`, `llk_math_eltwise_unary_sfpu_max.h`) with template functions for each variant. These files were nearly identical except for the op-specific type, function pointer and kernel include. This led to duplicated code and compilation overhead.

**Now, you should use the macros in `llk_math_eltwise_unary_sfpu_macros.h` directly from your API header (e.g., `eltwise_unary/eqz.h`, `eltwise_unary/max.h`).**

## Standard Template Approach

Instead of creating a new `llk_math_eltwise_unary_sfpu_<op>.h` for each op, use the macros to generate the required init and compute functions. The macros handle the op type, function pointer, and kernel include for you.

## Step-by-Step: Adding a New Op

1. **Implement your op in the low-level kernel (e.g., `ckernel_sfpu_<op>.h`).**
2. **In your API header (e.g., `eltwise_unary/<op>.h`), include both the macro header and the specific kernel header.**
3. **Use the macros to define your init and compute functions.**
4. **Add the include guard to `sfpu_split_includes.h`.**
5. **Register in `unary_op_utils.cpp` (macro definition, init/func, approx mode).**
