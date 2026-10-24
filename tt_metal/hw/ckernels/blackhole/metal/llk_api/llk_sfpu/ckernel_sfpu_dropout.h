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

// Mixes one salted lane draw and zeroes LREG6 where it falls below probability. Exactly
// DROPOUT_MASK_ROW_LEN instructions, recorded into the replay buffer by calculate_dropout.
inline void dropout_mask_row() {
    ////////////////////////
    // LREG4 = mix(draw + salt); also reads the next raw draw into LREG0.
    // Unset sign-bit for easy comparison with probability
    ////////////////////////
    finish_mix_uint32_mul24();
    TTI_SFPSETSGN(0, p_sfpu::LREG4, p_sfpu::LREG4, 1);

    ////////////////////////
    // Drop samples
    // v_if (rand < probability)
    //   dst_reg[0] = 0.0f;
    ///////////////////////
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG4, 10);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    // Salt the next draw and start its mixer; harmless after the final row.
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
    begin_mix_uint32_mul24();
}

// finish_mix_uint32_mul24 (10) + SFPSETSGN, SFPIADD, SFPMOV, SFPENCC, SFPIADD, begin_mix (1).
// Fits the SFPU half of the replay buffer (slots [0, math::replay_buf_offset)).
constexpr std::uint32_t DROPOUT_MASK_ROW_LEN = 16;

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(uint probability, uint scale) {
    // SFPU microcode
    //
    // Register map:
    //   LREG0 = raw lane PRNG draw (+ lane salt), mixer input
    //   LREG1 = scale
    //   LREG2 = probability
    //   LREG3 = lane salt (make_lane_salt)
    //   LREG4 = mixed draw
    //   LREG5 = mixer scratch
    //   LREG6 = sample
    //
    // The lane PRNGs are shifted views of one sequence, so their raw bits must not be
    // compared against probability directly (tt-llk#1701 item 8). Every draw goes through
    // the same lane salt and bijective finalizer that rand_tile uses (ckernel_sfpu_rand.h)
    // before the compare; see make_lane_salt / finish_mix_uint32_mul24 for the rationale.

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);

    // Prime the mixer with the first draw; finish_mix_uint32_mul24 reads the next one.
    make_lane_salt();
    rand_prng<p_sfpu::LREG0>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
    begin_mix_uint32_mul24();

    ////////////////////////
    // Scale samples
    // dst_reg[0] = dst_reg[0] * sFloat16b(scale);
    ///////////////////////
    // The first row records the mask row into the replay buffer and executes it; the
    // remaining rows replay it.
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, 3, 0);
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    load_replay_buf<Exec>(0, DROPOUT_MASK_ROW_LEN, [] { dropout_mask_row(); });
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, 3, 0);
    sfpi::dst_reg++;
#pragma GCC unroll 7
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        lltt::replay(0, DROPOUT_MASK_ROW_LEN);
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
