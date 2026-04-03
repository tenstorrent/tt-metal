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

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    init_prng_seed(seed);
}

// RReLU(x) = x if x >= 0
//            a*x if x < 0
// Eval mode (training_uint == 0): a = lower + range * 0.5 = (lower + upper) / 2
// Training mode (training_uint != 0): a ~ Uniform(lower, upper) per element
//
// Parameters:
//   lower_uint: bitcast of lower bound (float)
//   range_uint: bitcast of (upper - lower) (float), precomputed on host
//   training_uint: bitcast of 1.0f if training, 0 if eval
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_uint, uint range_uint, uint training_uint) {
    // training_uint is bitcast of 1.0f (= 0x3f800000) for training, 0 for eval
    if (training_uint != 0) {
        // ---- Training mode: random slope per element ----
        // Uses raw TTI instructions for PRNG access (same pattern as rand + leaky_relu).

        // Load range into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, range_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, range_uint >> 16);

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_uint >> 16);

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Generate random float in [0, 1) using PRNG (same technique as ckernel_sfpu_rand.h)
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // slope = rand_01 * range + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);

            // Set CC: lanes where input < 0 become active
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

            // CC-guarded multiply: input *= slope (only for negative lanes)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

            // Reset CC: all lanes active
            TTI_SFPENCC(0, 0, 0, 0);

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, 3, 0);

            dst_reg++;
        }
    } else {
        // ---- Eval mode: fixed slope = (lower + upper) / 2 = lower + range * 0.5 ----
        // Uses SFPI abstractions (same pattern as prelu).
        vFloat lower_val = Converter::as_float(lower_uint);
        vFloat range_val = Converter::as_float(range_uint);
        vFloat slope = lower_val + range_val * vFloat(0.5f);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat v = dst_reg[0];
            v_if(v < 0.0f) { v = v * slope; }
            v_endif;
            dst_reg[0] = v;
            dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
