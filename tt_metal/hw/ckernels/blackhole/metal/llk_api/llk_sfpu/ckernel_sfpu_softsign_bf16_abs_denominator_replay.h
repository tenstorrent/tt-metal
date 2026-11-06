// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::softsign {
using namespace ::sfpi;
using ::sfpi::DataLayout;
inline void abs_denominator_loadmacro_init() {
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

template <uint32_t Hold, uint32_t Advance, bool LoadMacro = true, uint32_t BoundBits = 0u>
inline constexpr uint32_t abs_denominator_bh_body_head_slots = 15u;
template <uint32_t Hold, uint32_t Advance, bool LoadMacro = true, uint32_t BoundBits = 0u>
inline void abs_denominator_bh_body_head() {
    constexpr uint32_t bound = BoundBits == 0u ? ckernel::p_sfpu::LREG13 : ckernel::p_sfpu::LREG6;
    static_assert((BoundBits & 0xffffu) == 0u, "the row loads a BF16-exact bound in one SFPLOADI");
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);  // x
    if constexpr (BoundBits == 0u) {
        TTI_SFPNOP;  // conservative replay load-consumer gap
    }
    if constexpr (BoundBits != 0u) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, SFPLOADI_MOD0_FLOATB, BoundBits >> 16);  // bound; load-consumer gap
    }
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                  1);  // magnitude = |x|
    TTI_SFPSWAP(0, bound, ckernel::p_sfpu::LREG1,
                sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // magnitude=min(|x|, bound)
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // den = magnitude + 1.0
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 0);
    TTI_SFPARECIP(0, 7, 2, 0);  // reciprocal seed -> L2
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                  0);  // restore x sign; reciprocal-seed hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG3,
        2);  // t = den*r0 - 2.0
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG0,
        0);  // finite identity / exponent-FF poison fills residual hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0);  // CC: first residual < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        1);  // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
}
template <uint32_t Hold, uint32_t Advance, bool LoadMacro = true, uint32_t BoundBits = 0u>
inline void abs_denominator_bh_body_tail() {
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        2);                    // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // CC-balanced replay body
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
    if constexpr (LoadMacro) {
        TTI_SFPLOADMACRO((0 << 2) | (ckernel::p_sfpu::LREG4 & 3), 0, Advance, (ckernel::p_sfpu::LREG4 >> 2));
    } else {
        // Nearest to BF16, stored over x.
        TTI_SFPNOP;  // result-MUL hazard gap
        TTI_SFP_STOCH_RND(
            SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
    }
}
template <uint32_t Hold, uint32_t Advance, uint32_t BoundBits = 0u>
inline void abs_denominator_bh_body() {
    abs_denominator_bh_body_head<Hold, Advance, false, BoundBits>();
    abs_denominator_bh_body_tail<Hold, Advance, false, BoundBits>();
}
}  // namespace ckernel::sfpu::ttpoly::softsign
