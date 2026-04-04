# Reference Analysis: selu (SELU)

## Overview
selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1))). Uses exponential computation internally.

## Key Patterns

### Exponential Init
```cpp
template <bool APPROXIMATION_MODE>
inline void selu_init() {
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}
```

### Exponential Computation
```cpp
sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
    v, exp_base_scale_factor);
```

### Includes
```cpp
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
```

## Relevance to rpow
rpow = base^x = exp(x * ln(base)). While we could implement rpow using exp, the power algorithm from `power` is more accurate and handles edge cases better. The selu pattern shows how to use exponential functions if needed as a fallback.
