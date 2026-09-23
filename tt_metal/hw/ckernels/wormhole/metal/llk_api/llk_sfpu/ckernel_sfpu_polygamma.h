#pragma once

#include "ckernel_sfpu_common.h"
#include "ckernel_sfpu_power.h"

// Polygamma SFPU kernel implementation for Wormhole architecture.
// This kernel mirrors the Blackhole implementation and includes the same
// correction to fold the factorial scale into the accumulation to avoid
// fp32 subnormal underflow for higher orders.

namespace sfpi {

inline vFloat polygamma(vFloat x, int n) {
    TT_FATAL(n >= 1 && n <= 11, "polygamma order n must be between 1 and 11");

    const float sign = (n % 2 == 0) ? -1.0f : 1.0f;
    float factorial = 1.0f;
    for (int i = 2; i <= n; ++i) {
        factorial *= static_cast<float>(i);
    }
    const float scale = sign * factorial;

    vFloat inv_x = recip(x);
    vFloat inv_power = power(inv_x, n, inv_x * scale);

    vFloat sum = inv_power;
    const int NUM_TERMS = 6;
    for (int k = 1; k < NUM_TERMS; ++k) {
        const float binom = static_cast<float>(tg::binomial_coefficient(n + k, k));
        inv_power = inv_power * inv_x;
        vFloat term = inv_power * binom;
        if ((k & 1) == 1) {
            term = -term;
        }
        sum = sum + term;
    }

    const float tail_coeffs[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // actual coefficients generated at compile time
    vFloat inv_z = recip(x + static_cast<float>(NUM_TERMS));
    vFloat inv_z2 = inv_z * inv_z;
    vFloat tail = vFloat(0.0f);
    if (n % 2 == 0) {
        vFloat inv_z_power = power(inv_z2, n / 2, inv_z2 * scale);
        for (int i = 0; i < 5; ++i) {
            tail = tail + inv_z_power * vFloat(tail_coeffs[i]);
        }
    } else {
        vFloat inv_z_power = power(inv_z, n, inv_z * scale);
        for (int i = 0; i < 5; ++i) {
            tail = tail + inv_z_power * vFloat(tail_coeffs[i]);
        }
    }

    sum = sum + tail;
    vFloat result = sum * vFloat(1.0f);
    return result;
}

} // namespace sfpi
