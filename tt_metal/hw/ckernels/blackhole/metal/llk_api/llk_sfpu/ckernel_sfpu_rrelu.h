// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

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
        sfpi::vFloat lower_v = Converter::as_float(param0);
        sfpi::vFloat upper_v = Converter::as_float(param1);
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
        // Strategy:
        // 1. Generate random uint32 via PRNG hardware
        // 2. Force exponent to 127 and clear sign bit -> float in [1.0, 2.0)
        // 3. Compute a = rand_1_2 * range + offset
        //    where range = upper - lower, offset = 2*lower - upper
        //    This maps [1.0, 2.0) -> [lower, upper)
        // 4. For x < 0: result = x * a; else result = x
        //
        // Uses raw TTI instructions for PRNG and bit manipulation.

        // Seed PRNG once (static guard)
        static bool prng_ready = false;
        if (!prng_ready) {
            init_prng_seed(0xDEADBEEF);
            prng_ready = true;
        }

        // Precompute range and offset from lower/upper
        float lower_f = Converter::as_float(param0);
        float upper_f = Converter::as_float(param1);
        float range_f = upper_f - lower_f;
        float offset_f = 2.0f * lower_f - upper_f;

        union {
            float f;
            std::uint32_t u;
        } range_bits{range_f}, offset_bits{offset_f};

        // Load offset into LREG1 (src_c for SFPMAD: a = rand * range + offset)
        TT_SFPLOADI(p_sfpu::LREG1, 10, offset_bits.u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, offset_bits.u >> 16);

        // Load range into LREG2 (src_b for SFPMAD: rand * range)
        TT_SFPLOADI(p_sfpu::LREG2, 10, range_bits.u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, range_bits.u >> 16);

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            // Load x from DEST into LREG0
            TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);

            // Generate random uint32 per lane via PRNG hardware -> LREG3
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            // Force exponent to 127: float in [1.0, 2.0) with random mantissa
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            // Clear sign bit to ensure positive
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

            // a = rand_1_2 * range + offset
            // LREG3 = LREG3 * LREG2 + LREG1
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, 0);

            // Conditionally apply: if x < 0, result = x * a; else result = x
            // Set CC where LREG0 < 0
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);

            // CC-guarded: LREG0 = LREG0 * LREG3 + 0 = x * a (only for x < 0)
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

            // End conditional context
            TTI_SFPENCC(0, 0, 0, 0);

            // Store result back to DEST
            TTI_SFPSTORE(0, 0, 3, 0);

            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
