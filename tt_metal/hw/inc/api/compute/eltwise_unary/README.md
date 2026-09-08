# How to Add a New Eltwise Unary Operator Using Macros

This document describes the recommended approach for adding a new eltwise unary operator in `tt-metal` using the macro system introduced in `llk_math_eltwise_unary_sfpu_macros.h`. This replaces the previous pattern of intermediate function wrappers and per-op header files.

## Why Macros?

Previously, each unary op (e.g., eqz, log1p, max, negative) required a dedicated intermediate header (e.g., `llk_math_eltwise_unary_sfpu_eqz.h`, `llk_math_eltwise_unary_sfpu_max.h`) with template functions for each variant. These files were nearly identical except for the op-specific type, function pointer and kernel include. This led to duplicated code and compilation overhead.

**Now, you should use the macros in `llk_math_eltwise_unary_sfpu_macros.h` directly from your API header (e.g., `eltwise_unary/eqz.h`, `eltwise_unary/max.h`).**

## Dest-acc template: only where it is used

An API gets `template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>` **only if dest-acc is forwarded into a used template of the thing it instantiates**:

1. SFPU execute: dest-acc appears in the `TEMPLATES` tuple of `SFPU_UNARY_CALL` / `SFPU_BINARY_CALL` / `SFPU_TERNARY_CALL`
2. SFPU init: dest-acc is passed to `SFPU_UNARY_INIT_ACCUM` or to `SFPU_UNARY_INIT_FN` / `_FN_ARGS` callback templates
3. Non-SFPU LLK: dest-acc is a template argument of the unpack/math/pack LLK the API instantiates

If none of those hold, the API stays untemplated. Dest-acc-independent ops still work after `set_fp32_dest_acc` because hardware mode is already programmed and WH/BH dest bounds are runtime.

Do **not** add `is_fp32_dest_acc_en` just because `SFPU_*_CALL` used to take a dest-acc argument. That slot is gone; dest-acc lives in `TEMPLATES` when the kernel needs it.

Keep the default `= DST_ACCUM_MODE` on dest-acc-sensitive APIs so existing callers stay valid.

## Standard Template Approach

Instead of creating a new `llk_math_eltwise_unary_sfpu_<op>.h` for each op, use the macros to generate the required init and compute functions. The macros handle the op type, function pointer, and kernel include for you.

### Example: Dest-acc-independent op (e.g., negative)

The kernel does not branch on dest-acc, so the API has no dest-acc template.

```cpp
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_negative.h"

namespace ckernel {
    ALWI void negative_tile_init() {
        MATH(SFPU_UNARY_INIT(negative));
    }
    ALWI void negative_tile(uint32_t idst) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, _calculate_negative_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
    }
}
```

### Example: Dest-acc-sensitive op (e.g., tanh)

The kernel takes `is_fp32_dest_acc_en` in its templates, so the API does too and forwards it into `TEMPLATES` / the init callback.

```cpp
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel {
    template <bool fast_and_approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
    ALWI void tanh_tile_init() {
        MATH(SFPU_UNARY_INIT_FN(tanh, sfpu::tanh_init, (fast_and_approx, is_fp32_dest_acc_en)));
    }
    template <bool fast_and_approx = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
    ALWI void tanh_tile(uint32_t idst) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            calculate_tanh,
            (fast_and_approx, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
}
```

After `set_fp32_dest_acc<true>()`, call dest-acc-sensitive ops with the active mode (`tanh_tile<false, true>(0)`) and dest-acc-independent ops with no extra template (`negative_tile(0)`, `binary_max_tile(idst0, idst1, odst)`).

Bare inits that depend on dest-acc (`asin`/`acos`/`softcap` on Blackhole) should use `SFPU_UNARY_INIT_ACCUM(op, is_fp32_dest_acc_en)` when the active dest mode may differ from `DST_ACCUM_MODE` (e.g. after `set_fp32_dest_acc`).

## Step-by-Step: Adding a New Op

1. **Implement your op in the low-level kernel (e.g., `ckernel_sfpu_<op>.h`).**
2. **In your API header (e.g., `eltwise_unary/<op>.h`), include both the macro header and the specific kernel header:**
   - `#include "llk_math_eltwise_unary_sfpu_macros.h"`
   - `#include "ckernel_sfpu_<op>.h"` (only include the kernel header needed for your op)
3. **Use the macros to define your init and compute functions.**
   - Choose the macro that matches your op's requirements or, if your op requires a new macro, add it to `llk_math_eltwise_unary_sfpu_macros.h` and document it.
   - Pass the op name, type, and any required parameters.
   - Add `is_fp32_dest_acc_en` only if the kernel, init, or LLK actually uses it (see rule above).

## Migration Notes

- **Do not create new `llk_math_eltwise_unary_sfpu_<op>.h` files.**
- **Remove old intermediate headers if you find them.**
- **Use the macros in your API headers for all new and migrated ops.**
- **Refer to `llk_math_eltwise_unary_sfpu_macros.h` for macro documentation and argument order.**

## FAQ

**Q: What if my op needs a special function pointer or extra runtime parameters?**

A: Use the most specific macro available or add a new one if needed and document it

**Q: How do I migrate an old op to the new macro system?**

A: Remove the intermediate header, replace the function calls with the appropriate macro.

**Q: Does every SFPU op need `is_fp32_dest_acc_en`?**

A: No. Only ops whose `calculate_*` / init / LLK templates actually branch on dest-acc. Independent ops (abs, identity, relu, bitwise, comparisons, ...) stay untemplated.
