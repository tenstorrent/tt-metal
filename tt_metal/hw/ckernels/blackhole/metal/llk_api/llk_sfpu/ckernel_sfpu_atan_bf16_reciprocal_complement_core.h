// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::atan {
using namespace ::sfpi;
using ::sfpi::DataLayout;
constexpr unsigned reciprocal_complement_held_slots = 4u;
// Below degree eight, rung J feeds coefficient kDegree - 2 - J: LREG7/6/5 first, then LREG3, then
// LCONST_0 for the zero constant term.
template <class Config, unsigned J>
constexpr unsigned reciprocal_complement_short_addend() {
    if constexpr (Config::kDegree - 2u == J) {
        return 9;  // admission requires c0 == 0
    } else if constexpr (J < 3) {
        return 7u - J;
    } else {
        return 3;
    }
}
template <class Config, unsigned J>
inline void reciprocal_complement_short_rung() {
    constexpr unsigned kLateLoad = Config::kDegree - 5u;
    if constexpr ((J == 0 || J == 1) && Config::kDegree >= 6) {
        constexpr unsigned word = Config::kCoefficientBits[kLateLoad];
        if constexpr (J == 0) {
            TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, word >> 16);
        } else {
            TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, word & 0xffffu);
        }
    } else if constexpr (J == 4 && Config::kDegree == 7) {
        TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
        TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(2, 1, (reciprocal_complement_short_addend<Config, J>()), 2, 0);
    if constexpr (J + 2u < Config::kDegree) {
        reciprocal_complement_short_rung<Config, J + 1u>();
    }
}
template <class Config, unsigned Hold, unsigned Advance, bool LoadMacro = true, bool Held = false>
inline constexpr uint32_t reciprocal_complement_body_head_slots = 11u + ((Held) ? 5u : 1u);
template <class Config, unsigned Hold, unsigned Advance, bool LoadMacro = true>
inline void reciprocal_complement_body_head() {
    static_assert(Config::kDegree == 4, "the degree-eight row has no free replay slot");
    TTI_SFPLOAD(0, 0, Hold, 0);
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, 0, 1, 1);
    TTI_SFPARECIP(0, 1, 3, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, 1, ckernel::p_sfpu::LCONST_1, 0, 0);
    TTI_SFPMAD(1, 3, 12, 2, 2);
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 2, 0, 0);
    TTI_SFPMAD(2, 3, ckernel::p_sfpu::LCONST_0, 3, 3);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSWAP(0, 3, 1, 1);
    TTI_SFPLOADI(2, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree] >> 16);
    TTI_SFPLOADI(2, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree] & 0xffffu);
    TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 1] >> 16);
    TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 1] & 0xffffu);
    TTI_SFPMAD(2, 1, 3, 2, 0);
}
template <class Config, unsigned Hold, unsigned Advance, bool LoadMacro = true>
inline void reciprocal_complement_body_tail() {
    reciprocal_complement_short_rung<Config, 0>();
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, 2, 4, 2, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    if constexpr (LoadMacro) {
        TTI_SFPLOADMACRO(3, 0, Advance, 0);
    } else {
        // x's sign onto the result, Nearest to BF16, stored over x.
        TTI_SFPLOAD(3, 0, Hold, 0);
        TTI_SFPNOP;
        TTI_SFPSETSGN(0, 2, 3, 0);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, 3, 3, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(3, 0, Advance, 0);
    }
}
template <class Config, unsigned Hold, unsigned Advance>
inline void reciprocal_complement_body() {
    reciprocal_complement_body_head<Config, Hold, Advance, false>();
    reciprocal_complement_body_tail<Config, Hold, Advance, false>();
}
}  // namespace ckernel::sfpu::ttpoly::atan
