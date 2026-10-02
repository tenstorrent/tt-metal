// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_rand.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(uint probability, uint scale) {
    // SFPU microcode

    make_lane_salt();

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * sFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);

        // Salt and mix the shifted per-lane hardware streams before thresholding.
        rand_prng<p_sfpu::LREG0>();
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        begin_mix_uint32_mul24();
        finish_mix_uint32_mul24<false>();
        TTI_SFPSETSGN(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpsetsgn_mod1_arg_imm);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = 0.0f;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG4, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, 3, 0);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void dropout_init(const uint seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    init_prng_seed(seed);
}

}  // namespace sfpu
}  // namespace ckernel
