#ifndef CKERNEL_SFPU_RECIP_H
#define CKERNEL_SFPU_RECIP_H

#include "ckernel_sfpu_common.h"

namespace ckernel {

namespace sfpu {

template <typename T>
inline T reciprocal(T x) {
    // FP32 reciprocal implementation (unchanged)
    if constexpr (std::is_same_v<T, float>) {
        // Existing FP32 reciprocal code
        float x_abs = fabsf(x);
        if (x_abs == 0.0f) {
            return copysignf(INFINITY, x);
        } else if (isinf(x_abs)) {
            return copysignf(0.0f, x);
        } else if (isnan(x)) {
            return x;
        }
        
        // Faithfully rounded FP32 reciprocal
        float recip = 1.0f / x;
        return recip;
    }
    // BF16 reciprocal implementation (updated for correct rounding)
    else if constexpr (std::is_same_v<T, bfloat16>) {
        // Handle special cases
        if (x == bfloat16(0.0f)) {
            return bfloat16(copysignf(INFINITY, x.value()));
        } else if (isinf(x.value())) {
            return bfloat16(copysignf(0.0f, x.value()));
        } else if (isnan(x.value())) {
            return x;
        }
        
        // Correctly rounded BF16 reciprocal
        float x_float = static_cast<float>(x);
        float recip_float = 1.0f / x_float;
        bfloat16 recip_bf16 = bfloat16(recip_float);
        
        // Check if the result is correctly rounded
        float next_down = nextafterf(recip_float, -INFINITY);
        float next_up = nextafterf(recip_float, INFINITY);
        
        if (fabsf(recip_float - x_float * static_cast<float>(recip_bf16)) >
            fabsf(next_down - x_float * static_cast<float>(bfloat16(next_down)))) {
            recip_bf16 = bfloat16(next_down);
        } else if (fabsf(recip_float - x_float * static_cast<float>(recip_bf16)) >
                     fabsf(next_up - x_float * static_cast<float>(bfloat16(next_up)))) {
            recip_bf16 = bfloat16(next_up);
        }
        
        return recip_bf16;
    } else {
        static_assert(always_false<T>::value, "Unsupported type for reciprocal operation");
    }
}

} // namespace sfpu

} // namespace ckernel

#endif // CKERNEL_SFPU_RECIP_H
