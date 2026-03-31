// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u, uint seed) {
    if (seed == 0) {
        // Evaluation mode: use deterministic midpoint slope = (lower + upper) / 2
        vFloat lower = Converter::as_float(lower_u);
        vFloat upper = Converter::as_float(upper_u);
        vFloat midpoint = (lower + upper) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat a = dst_reg[0];
            v_if(a < 0.0f) { a = a * midpoint; }
            v_endif;
            dst_reg[0] = a;
            dst_reg++;
        }
    } else {
        // Training mode: per-element random slopes in [lower, upper)
        // Register allocation:
        //   LREG1 = scale = upper - lower
        //   LREG2 = lower
        //   LREG0 = working (input / output)
        //   LREG3 = working (random slope)

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_u >> 16);

        // Load upper into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, upper_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, upper_u >> 16);

        // Compute scale = upper - lower: LREG1 = LREG1 * 1.0 + (-lower)
        // Load -lower into LREG3 (flip sign bit in upper 16 bits)
        TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG3, 8, (lower_u >> 16) ^ 0x8000);

        // LREG1 = LREG1(upper) * LCONST_1(1.0) + LREG3(-lower) = upper - lower
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG1, 0);

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);

            // Generate random float in [0, 1) using hardware PRNG
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            // Clear sign bit (force positive)
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Set exponent to 127 -> value in [1.0, 2.0)
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Subtract 1.0 to get [0.0, 1.0)
            TTI_SFPADDI(0xbf80 /*-1.0f in bfloat16*/, p_sfpu::LREG3, 0);

            // Scale to [lower, upper): slope = rand * scale + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);

            // Apply slope only to negative elements
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                          // CC <- (LREG0 < 0)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // x * slope (CC-guarded)
            TTI_SFPENCC(0, 0, 0, 0);                                                       // CC <- ALL_ENABLED

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    if (seed != 0) {
        init_prng_seed(seed);
    }
}

}  // namespace sfpu
}  // namespace ckernel
