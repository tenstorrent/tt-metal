// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected reciprocal coordinate and reconstruction. Callers retain terminals,
// coefficient ownership, rounding and traversal.
namespace sfpi {
template <class Reciprocal>
inline vFloat reciprocal_complement_reduce(vFloat original, vFloat& reduced, Reciprocal reciprocal) {
    vFloat abs_x = setsgn(original, 0);
    vFloat inverse = reciprocal(abs_x);
    reduced = abs_x;
    v_if(abs_x > 1.0f) { reduced = inverse; }
    v_endif;
    return reduced * reduced;
}
inline vFloat reciprocal_complement_expand(vFloat original, vFloat reduced, vFloat result) {
    constexpr float complement = 1.5707963267948966f;
    vFloat core_value = reduced * result;
    vFloat magnitude = core_value;
    v_if(setsgn(original, 0) > 1.0f) { magnitude = complement - core_value; }
    v_endif;
    return copysgn(magnitude, original);
}
#if defined(ARCH_BLACKHOLE)
template <class Config, unsigned J>
constexpr unsigned reciprocal_complement_addend() {
    if constexpr (J == 6 && Config::kCoefficientBits[0] == 0u) {
        return 9;
    }
    if constexpr (J == 0 || J == 4 || J == 5 || J == 6) {
        return 3;
    }
    return 8u - J;
}
template <class Config, unsigned J>
inline void reciprocal_complement_rung() {
    if constexpr (reciprocal_complement_addend<Config, J>() == 3) {
        if constexpr (J == 4) {
            TTI_SFPNOP;
        } else {
            TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[6 - J] >> 16);
            TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[6 - J] & 0xffffu);
        }
    } else if constexpr (J == 1) {
        TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[2] >> 16);
    } else if constexpr (J == 2) {
        TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[2] & 0xffffu);
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(2, 1, (reciprocal_complement_addend<Config, J>()), 2, 0);
}
template <class Config, unsigned Hold, unsigned Advance>
inline void reciprocal_complement_body() {
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
    TTI_SFPMAD(13, 1, 14, 2, 0);
    reciprocal_complement_rung<Config, 0>();
    reciprocal_complement_rung<Config, 1>();
    reciprocal_complement_rung<Config, 2>();
    reciprocal_complement_rung<Config, 3>();
    reciprocal_complement_rung<Config, 4>();
    reciprocal_complement_rung<Config, 5>();
    reciprocal_complement_rung<Config, 6>();
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, 2, 4, 2, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPLOADMACRO(3, 0, Advance, 0);
}
#endif
}  // namespace sfpi
