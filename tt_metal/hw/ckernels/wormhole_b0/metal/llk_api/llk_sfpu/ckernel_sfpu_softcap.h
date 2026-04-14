// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// tanh(u) = (E - 1)/(E + 1) where E = 2^(2|u|*log2(e))
// E computed via: floor/frac range reduction + degree-10 polynomial for 2^frac
// Newton-Raphson reciprocal for the division.
// Taylor degree-17 for small |u| to avoid cancellation.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(float cap, float inv_cap) {
    constexpr float two_log2e = 2.8853900817779268f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat au = sfpi::abs(u);

        // y = 2|u| * log2(e); clamp to prevent overflow
        sfpi::vFloat y = au * two_log2e;
        v_if(y > 125.0f) { y = 125.0f; }
        v_endif;

        // Floor via add-subtract 2^23 trick (round-to-nearest, then adjust)
        sfpi::vFloat k = (y + 8388608.0f) - 8388608.0f;
        v_if(k > y) { k = k - sfpi::vConst1; }
        v_endif;

        sfpi::vFloat r = y - k;  // fractional part in [0, 1)

        // 2^r via degree-12 polynomial (Taylor of exp(r*ln2))
        constexpr float c12 = 1.2352386e-10f;  // (ln2)^12/12!
        constexpr float c11 = 1.3534455e-9f;   // (ln2)^11/11!
        constexpr float c10 = 2.4022651e-8f;   // (ln2)^10/10!
        constexpr float c9 = 2.0908898e-7f;    // (ln2)^9/9!
        constexpr float c8 = 1.5252734e-6f;    // (ln2)^8/8!
        constexpr float c7 = 1.3215487e-5f;    // (ln2)^7/7!
        constexpr float c6 = 1.5252734e-4f;    // (ln2)^6/6!
        constexpr float c5 = 1.3333558e-3f;    // (ln2)^5/5!
        constexpr float c4 = 9.6181291e-3f;    // (ln2)^4/4!
        constexpr float c3 = 5.5504109e-2f;    // (ln2)^3/3!
        constexpr float c2 = 2.4022651e-1f;    // (ln2)^2/2!
        constexpr float c1 = 6.9314718e-1f;    // ln2

        sfpi::vFloat pw = c12;
        pw = pw * r + c11;
        pw = pw * r + c10;
        pw = pw * r + c9;
        pw = pw * r + c8;
        pw = pw * r + c7;
        pw = pw * r + c6;
        pw = pw * r + c5;
        pw = pw * r + c4;
        pw = pw * r + c3;
        pw = pw * r + c2;
        pw = pw * r + c1;
        pw = pw * r + sfpi::vConst1;  // pw = 2^r, in [1, 2)

        // E = 2^k * pw: multiply pw by 2^k via exponent set
        // pw is in [1, 2) with biased exponent 127. Set to 127 + k.
        sfpi::vInt ki = sfpi::reinterpret<sfpi::vInt>(k + 8388608.0f) & sfpi::vInt(0x7FFFFF);
        sfpi::vFloat E = sfpi::setexp(pw, sfpi::vUInt(127 + ki));

        // tanh(|u|) = (E - 1) / (E + 1) via NR reciprocal
        sfpi::vFloat den = E + sfpi::vConst1;
        sfpi::vFloat rv = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(0x7EF311C7) - sfpi::reinterpret<sfpi::vInt>(den));
        rv = rv * (2.0f - den * rv);
        rv = rv * (2.0f - den * rv);
        rv = rv * (2.0f - den * rv);
        sfpi::vFloat tv = (E - sfpi::vConst1) * rv;

        // Small |u|: Taylor degree-17
        v_if(au < 0.5f) {
            sfpi::vFloat u2 = au * au;
            sfpi::vFloat tp = u2 * (-0.0014560099f) + 0.0035921280f;
            tp = tp * u2 + (-0.0088632360f);
            tp = tp * u2 + 0.0218694890f;
            tp = tp * u2 + (-0.0539682540f);
            tp = tp * u2 + 0.1333333400f;
            tp = tp * u2 + (-0.3333333400f);
            tp = tp * u2 + sfpi::vConst1;
            tv = au * tp;
        }
        v_endif;

        // Saturate
        v_if(au >= 9.0f) { tv = sfpi::vConst1; }
        v_endif;

        // Apply sign
        sfpi::vFloat result = tv;
        v_if(u < 0.0f) { result = -tv; }
        v_endif;

        sfpi::dst_reg[0] = result * cap;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace ckernel::sfpu
