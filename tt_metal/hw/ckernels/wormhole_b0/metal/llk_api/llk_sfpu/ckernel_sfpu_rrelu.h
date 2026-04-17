// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace {
inline float uint32_to_float(std::uint32_t value) {
    union { std::uint32_t u; float f; } conv{value};
    return conv.f;
}
}  // namespace

// Randomized Leaky ReLU (RReLU):
//   f(x) = x              when x >= 0
//   f(x) = a * x          when x < 0
//
// Eval mode (param2 == 0):  a = (lower + upper) / 2
// Train mode (param2 != 0): a ~ Uniform(lower, upper) per element
//
// param0 = lower (bit-cast float as uint32)
// param1 = upper (bit-cast float as uint32)
// param2 = training flag (0 = eval, nonzero = train)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    if (param2 == 0) {
        // EVAL MODE: fixed slope = (lower + upper) / 2
        sfpi::vFloat lower_v = uint32_to_float(param0);
        sfpi::vFloat upper_v = uint32_to_float(param1);
        sfpi::vFloat slope = (lower_v + upper_v) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            v_if(x < 0.0f) { x = x * slope; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    } else {
        // TRAINING MODE: random slope per element in [lower, upper]
        //
        // PRNG is seeded in llk_math_eltwise_unary_sfpu_rrelu_init via init_prng_seed.
        // TTI_SFPMOV(0, 9, dest, 8) advances the per-lane PRNG state, giving
        // independent random uint32 values per SFPU lane per iteration.
        //
        // Strategy:
        // 1. Generate random uint32 via PRNG hardware into LREG3
        // 2. Set exponent=127 and clear sign to get float in [1.0, 2.0)
        // 3. Compute a = rand_1_2 * range + offset  (maps [1,2) to [lower,upper))
        // 4. For x < 0: result = x * a; else result = x

        float lower_f = uint32_to_float(param0);
        float upper_f = uint32_to_float(param1);
        sfpi::vFloat range_v = upper_f - lower_f;
        sfpi::vFloat offset_v = 2.0f * lower_f - upper_f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            // Generate per-lane random uint32 via PRNG hardware into LREG3
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);

            // Read random bits from LREG3, set exponent=127 -> [1.0, 2.0), clear sign
            sfpi::vFloat rand_raw(sfpi::l_reg[sfpi::LRegs::LReg3]);
            sfpi::vFloat rand_1_2 = sfpi::setsgn(sfpi::setexp(rand_raw, 127u), 0);

            // Map [1.0, 2.0) to [lower, upper)
            sfpi::vFloat a = rand_1_2 * range_v + offset_v;

            v_if(x < 0.0f) { x = x * a; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
