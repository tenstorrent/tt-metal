// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Coefficients are from the rpow scalar log2 precomputation (Horner form):
//   P(m) = c0 + m * (c1 + m * (c2 + m * c3))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;
        sfpi::vFloat b = -x + sfpi::vConst1;

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);
        sfpi::vFloat ma = sfpi::setexp(a, 127);
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;
        pa = pa * ma + sfpi::vConstFloatPrgm1;
        pa = pa * ma + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa;

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);
        sfpi::vFloat mb = sfpi::setexp(b, 127);
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;
        pb = pb * mb + sfpi::vConstFloatPrgm1;
        pb = pb * mb + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb;

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691
}

}  // namespace ckernel::sfpu
