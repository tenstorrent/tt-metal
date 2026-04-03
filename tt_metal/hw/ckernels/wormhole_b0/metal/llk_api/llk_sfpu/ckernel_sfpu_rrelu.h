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

// RReLU(x) = x if x >= 0; a*x if x < 0
// a is randomly sampled from Uniform(lower, upper) per element using the hardware PRNG.
// The PRNG must be seeded via rrelu_init() before calling this function.
//
// Register usage in the main loop (raw TTI):
//   LREG0: current element from DEST (x), then result
//   LREG1: lower bound parameter (constant across loop)
//   LREG2: range = upper - lower (constant across loop)
//   LREG3: random value -> slope
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_param, uint upper_param) {
    // Load lower into LREG1 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG1, 10, lower_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, lower_param >> 16);

    // Load upper into LREG2 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG2, 10, upper_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, upper_param >> 16);

    // Compute range = upper - lower into LREG3, then move to LREG2.
    // Use SFPMAD: LREG3 = LREG1 * (-1.0) + LREG2 = upper - lower
    // Load -1.0 into LREG3 first, then do MAD.
    // -1.0 in FP32 = 0xBF800000. In BF16_B = 0xBF80.
    TTI_SFPLOADI(p_sfpu::LREG3, 10, 0x0000);  // LREG3.lo16 = 0
    TTI_SFPLOADI(p_sfpu::LREG3, 8, 0xBF80);   // LREG3.hi16 = 0xBF80 -> LREG3 = -1.0f

    // LREG3 = LREG1 * LREG3 + LREG2 = lower * (-1.0) + upper = upper - lower
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPNOP;  // pipeline hazard after SFPMAD on Wormhole

    // Move range from LREG3 to LREG2 (so LREG3 is free for the loop)
    // SFPMOV: VD = VC (with mod1=0, copies register)
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load current element from DEST into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // Generate random float in [0, 1) into LREG3
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);                   // LREG3 = PRNG value
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);    // sign = 0
        TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // exponent = 127 -> [1.0, 2.0)
        TTI_SFPADDI(0xbf80 /*-1.0f*/, p_sfpu::LREG3, 0);      // LREG3 -= 1.0 -> [0.0, 1.0)
        TTI_SFPNOP;                                           // pipeline hazard after SFPADDI on Wormhole

        // slope = rand_01 * range + lower = LREG3 * LREG2 + LREG1 -> LREG3
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPNOP;  // pipeline hazard after SFPMAD on Wormhole

        // Now LREG3 = slope in [lower, upper)
        // For negative elements: result = x * slope
        // Set CC based on sign of LREG0 (CC.Res = 1 if LREG0 < 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

        // CC-guarded: LREG0 = LREG0 * LREG3 + 0.0 (only for negative elements)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Clear CC result
        TTI_SFPENCC(0, 0, 0, 0);

        // Store result back to DEST
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
