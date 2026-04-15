// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// tanh(u) for |u| > 0.625: tanh = (1-em)/(1+em) where em = exp(-2|u|)
// 1/(1+em) computed via Newton-Raphson: 2 iterations from guess 0.6

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(const uint32_t param0, const uint32_t param1) {
    // Parameters are full float32 bit patterns passed as uint32_t compile-time constants
    sfpi::vFloat cap = __builtin_bit_cast(float, param0);
    sfpi::vFloat recip_cap = __builtin_bit_cast(float, param1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat au = sfpi::abs(x * recip_cap);

        // Compute exp(-2*au) via range reduction
        sfpi::vFloat v;
        v = (au + au) * 1.44269504089f + 0.5f;
        constexpr float MAGIC = 8388608.0f;
        sfpi::vFloat k = (v + MAGIC) - MAGIC;
        v = k * 0.69314718056f - au - au;  // r

        // exp(r) Horner degree 8 (for full fp32 accuracy)
        sfpi::vFloat em = 0.00002480159f;  // 1/40320
        em = em * v + 0.00019841270f;      // 1/5040
        em = em * v + 0.00138888889f;      // 1/720
        em = em * v + 0.00833333333f;      // 1/120
        em = em * v + 0.04166666667f;      // 1/24
        em = em * v + 0.16666666667f;      // 1/6
        em = em * v + 0.5f;                // 1/2
        em = em * v + 1.0f;                // 1
        em = em * v + 1.0f;                // constant

        // Multiply by 2^(-k) to get exp(-2*au)
        sfpi::vInt ki = sfpi::reinterpret<sfpi::vInt>(k + MAGIC);
        sfpi::vUInt kb = sfpi::reinterpret<sfpi::vUInt>(ki) & 0xFFu;
        sfpi::vUInt eb = sfpi::shft(127u - kb, 23);
        em = em * sfpi::reinterpret<sfpi::vFloat>(eb);

        // tanh = (1 - em) / (1 + em)
        // Compute 1/(1+em) via NR with initial guess 0.6
        // (for em in [0, 0.29], 1+em in [1, 1.29], 1/(1+em) in [0.78, 1])
        v = em + 1.0f;                     // denominator = 1+em
        sfpi::vFloat rcp = 1.0f - em;      // initial guess: 1-em ~ 1/(1+em) for small em
        rcp = rcp * (2.0f - v * rcp);      // NR iter 1
        rcp = rcp * (2.0f - v * rcp);      // NR iter 2
        rcp = rcp * (2.0f - v * rcp);      // NR iter 3
        rcp = rcp * (2.0f - v * rcp);      // NR iter 4

        sfpi::vFloat tanh_pos = (1.0f - em) * rcp;

        // For small |u| <= 0.55, override with Taylor
        v_if(au < 0.55f) {
            v = au * au;
            // For very small u, use fewer terms to minimize rounding accumulation
            v_if(au < 0.05f) {
                // 3 terms: tanh(u) ~ u*(1 - u^2/3 + 2u^4/15)
                tanh_pos = 0.13333333333f;
                tanh_pos = tanh_pos * v + (-0.33333333333f);
                tanh_pos = tanh_pos * v + 1.0f;
                tanh_pos = au * tanh_pos;
            }
            v_else {
                // 9 terms for moderate u
                tanh_pos = -0.00145583438f;
                tanh_pos = tanh_pos * v + 0.00359212803f;
                tanh_pos = tanh_pos * v + (-0.00886323553f);
                tanh_pos = tanh_pos * v + 0.02186948854f;
                tanh_pos = tanh_pos * v + (-0.05396825397f);
                tanh_pos = tanh_pos * v + 0.13333333333f;
                tanh_pos = tanh_pos * v + (-0.33333333333f);
                tanh_pos = tanh_pos * v + 1.0f;
                tanh_pos = au * tanh_pos;
            }
            v_endif;
        }
        v_endif;

        // Saturation
        v_if(au > 9.0f) { tanh_pos = sfpi::vConst1; }
        v_endif;

        // result = sign(x) * cap * tanh(|u|)
        v = cap * tanh_pos;
        v_if(x < 0.0f) { v = 0.0f - v; }
        v_endif;

        // For x == 0, result = 0
        v_if(x == 0.0f) { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
