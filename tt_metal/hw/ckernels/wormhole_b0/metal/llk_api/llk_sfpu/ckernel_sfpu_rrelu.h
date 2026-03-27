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

// RReLU eval/inference mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = slope * x      if x < 0
//   where slope = (lower + upper) / 2
//
// Parameters lower and upper are passed as bitcast uint32_t.
// The midpoint slope is computed on the SFPU.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u) {
    // Reconstruct float parameters from bitcast uint32_t
    vFloat lower = Converter::as_float(lower_u);
    vFloat upper = Converter::as_float(upper_u);

    // Compute slope = (lower + upper) * 0.5
    vFloat slope = (lower + upper) * vFloat(0.5f);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * slope; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

// RReLU training mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = a * x          if x < 0
//   where a ~ Uniform(lower, upper) is sampled per element
//
// Parameters: lower_u = bitcast(lower), range_u = bitcast(upper - lower)
// PRNG must be initialized via init_prng_seed() before calling this function.
// Uses raw TTI instructions for PRNG random number generation (same pattern as dropout/rand).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_training(uint lower_u, uint range_u) {
    // Load range = upper - lower into LREG2 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG2, 10, range_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, range_u >> 16);

    // Load lower into LREG3 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, 8, lower_u >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Step 1: Generate random float in [0, 1) -> LREG1
        // SFPMOV with lreg_c=9, instr_mod1=8 generates pseudorandom uint32
        TTI_SFPMOV(0, 9, p_sfpu::LREG1, 8);
        // Clear sign bit to ensure positive
        TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Set exponent to 127 -> float in [1, 2)
        TTI_SFPSETEXP(127, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Subtract 1 -> float in [0, 1)
        TTI_SFPADDI(0xbf80 /*-1.0f as fp16b*/, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 2: Scale to [lower, upper): slope = rand * range + lower
        // SFPMAD: LREG1 = LREG1 * LREG2 + LREG3
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 3: Load input from dst_reg -> LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

        // Step 4: Conditionally multiply negative inputs by random slope
        // SFPSETCC: sets condition code where LREG0 < 0 (sign bit check)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        // SFPMUL only applies to elements where CC is set (input < 0)
        // LREG0 = LREG0 * LREG1 + 0 (for negative elements only)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // End conditional block
        TTI_SFPENCC(0, 0, 0, 0);

        // Step 5: Store result back to dst_reg
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// PRNG seed initialization for training mode
template <bool APPROXIMATION_MODE>
inline void rrelu_init(const uint seed) {
    init_prng_seed(seed);
}

}  // namespace sfpu
}  // namespace ckernel
