// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
// exp(2u) via range-reduced degree-7 Taylor.
// Division via Newton-Raphson reciprocal (3 iterations).
// For |u| < 0.2: Taylor degree-7 for tanh (avoids cancellation in exp(2u)-1).
// For |u| > 9: tanh = ±1.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t cap_u, uint32_t rcap_u) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat u = sfpi::dst_reg[0] * sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(rcap_u));
        sfpi::vFloat ax = sfpi::setsgn(u, 0);

        // exp(2u) = 2^(2u * log2e) = 2^k * 2^f
        sfpi::vFloat z = ax * 2.8853900817779268f;
        sfpi::vFloat kf = z + 8388608.0f;
        kf = kf - 8388608.0f;
        sfpi::vFloat f = z - kf;
        // 2^f via degree-7 Taylor for exp(f*ln2)
        sfpi::vFloat g = f * 0.69314718055994530f;
        sfpi::vFloat e = g * 0.000198412698412698f + 0.001388888888888889f;
        e = e * g + 0.008333333333333333f;
        e = e * g + 0.041666666666666664f;
        e = e * g + 0.166666666666666660f;
        e = e * g + 0.5f;
        e = e * g + 1.0f;
        e = e * g + 1.0f;
        // Multiply by 2^k
        sfpi::vInt ki = sfpi::reinterpret<sfpi::vInt>(kf + 8388608.0f) - sfpi::vInt(0x4B000000);
        e = e * sfpi::setexp(sfpi::vConst1, sfpi::vUInt(127) + sfpi::reinterpret<sfpi::vUInt>(ki));

        // tanh = (e - 1) / (e + 1) via NR reciprocal
        sfpi::vFloat den = e + sfpi::vConst1;
        sfpi::vFloat r = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(0x7F000000) - sfpi::reinterpret<sfpi::vInt>(den));
        r = r * (sfpi::vFloat(2.0f) - den * r);
        r = r * (sfpi::vFloat(2.0f) - den * r);
        r = r * (sfpi::vFloat(2.0f) - den * r);
        sfpi::vFloat result = (e - sfpi::vConst1) * r;

        // Small |u|: Taylor degree-7 (avoids cancellation)
        v_if(ax < 0.22f) {
            sfpi::vFloat t = ax * ax;
            result = t * (-0.053968253968254f) + 0.133333333333333f;
            result = result * t + (-0.333333333333333f);
            result = result * t + 1.0f;
            result = result * ax;
        }
        v_endif;

        // Saturate
        v_if(ax > 9.0f) { result = sfpi::vFloat(1.0f); }
        v_endif;

        result = sfpi::setsgn(result, u);
        sfpi::dst_reg[0] = result * sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(cap_u));
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace ckernel::sfpu
