# How to Add a New Eltwise Unary Operator Using Macros

This document describes the recommended approach for adding a new elementwise unary operator in `tt-metal` using the macro system introduced in `llk_math_eltwise_unary_sfpu_macros.h`. This replaces the previous pattern of intermediate function wrappers and per-op header files.

## Why Macros?

Previously, each unary op (e.g., eqz, log1p, max, negative) required a dedicated intermediate header (e.g., `llk_math_eltwise_unary_sfpu_eqz.h`, `llk_math_eltwise_unary_sfpu_max.h`) with template functions for each variant. These files were nearly identical except for the op-specific type, function pointer and kernel include. This led to duplicated code and compilation overhead.

**Now, you should use the macros in `llk_math_eltwise_unary_sfpu_macros.h` directly from your API header (e.g., `eltwise_unary/eqz.h`, `eltwise_unary/max.h`).**

## Standard Template Approach

Instead of creating a new `llk_math_eltwise_unary_sfpu_<op>.h` for each op, use the macros to generate the required init and compute functions. The macros handle the op type, function pointer, and kernel include for you.

### Example: Adding a New Op (e.g., max)

```cpp
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_max.h"

namespace ckernel {
    // Init function for max
    template <bool fast_and_approx = true>
    ALWI void max_tile_init() {
        MATH(SFPU_UNARY_KERNEL_INIT(max, fast_and_approx));
    }
    // Compute function for max
    template <bool fast_and_approx = true>
    ALWI void max_tile(uint32_t idst) {
        MATH(SFPU_UNARY_NO_PARAM_KERNEL(max, RC, fast_and_approx, idst));
    }
}
```

### Example: Adding a New Op (e.g., negative)

```cpp
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_negative.h"

namespace ckernel {
    template <bool fast_and_approx = true>
    ALWI void negative_tile_init() {
        MATH(SFPU_UNARY_KERNEL_INIT(negative, fast_and_approx));
    }
    template <bool fast_and_approx = true>
    ALWI void negative_tile(uint32_t idst) {
        MATH(SFPU_UNARY_NO_PARAM_KERNEL(negative, RC, fast_and_approx, idst));
    }
}
```

## Step-by-Step: Adding a New Op

1. **Implement your op in the low-level kernel (e.g., `ckernel_sfpu_<op>.h`).**
2. **In your API header (e.g., `eltwise_unary/<op>.h`), include both the macro header and the specific kernel header:**
   - `#include "llk_math_eltwise_unary_sfpu_macros.h"`
   - `#include "ckernel_sfpu_<op>.h"` (only include the kernel header needed for your op)
3. **Use the macros to define your init and compute functions.**
   - Choose the macro that matches your op's requirements or, if your op requires a new macro, add it to `llk_math_eltwise_unary_sfpu_macros.h` and document it.
   - Pass the op name, type, and any required parameters.

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
