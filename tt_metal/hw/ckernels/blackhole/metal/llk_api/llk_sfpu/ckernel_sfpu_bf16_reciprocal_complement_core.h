// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"

// Selected reciprocal coordinate and reconstruction. Callers retain terminals,
// coefficient ownership, rounding and traversal.
namespace sfpi {
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
            ::ckernel::sfpu::bf16_sfpi::sfpnop();
        } else {
            ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[6 - J] >> 16);
            ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[6 - J] & 0xffffu);
        }
    } else if constexpr (J == 1) {
        ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[2] >> 16);
    } else if constexpr (J == 2) {
        ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[2] & 0xffffu);
    } else {
        ::ckernel::sfpu::bf16_sfpi::sfpnop();
    }
    ::ckernel::sfpu::bf16_sfpi::sfpmad(2, 1, (reciprocal_complement_addend<Config, J>()), 2, 0);
}
// Below degree eight, rung J feeds coefficient K = kDegree - 2 - J: the first three
// from LREG7/6/5, the next from LREG3 loaded in the gaps of rungs 0 and 1, a fifth
// through LREG3 again in its own rung, and a zero constant term from LCONST_0.
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
            ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_UPPER, word >> 16);
        } else {
            ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_LOWER, word & 0xffffu);
        }
    } else if constexpr (J == 4 && Config::kDegree == 7) {
        ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
        ::ckernel::sfpu::bf16_sfpi::sfploadi(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    } else {
        ::ckernel::sfpu::bf16_sfpi::sfpnop();
    }
    ::ckernel::sfpu::bf16_sfpi::sfpmad(2, 1, (reciprocal_complement_short_addend<Config, J>()), 2, 0);
    if constexpr (J + 2u < Config::kDegree) {
        reciprocal_complement_short_rung<Config, J + 1u>();
    }
}
template <class Config, unsigned Hold, unsigned Advance>
inline void reciprocal_complement_body() {
    ::ckernel::sfpu::bf16_sfpi::sfpload(0, 0, Hold, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpsetsgn(0, 0, 1, 1);
    ::ckernel::sfpu::bf16_sfpi::sfparecip(0, 1, 3, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpmad(ckernel::p_sfpu::LCONST_neg1, 1, ckernel::p_sfpu::LCONST_1, 0, 0);
    ::ckernel::sfpu::bf16_sfpi::sfpmad(1, 3, 12, 2, 2);
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, 2, 0, 0, cc_region);
        ::ckernel::sfpu::bf16_sfpi::sfpmad(2, 3, ckernel::p_sfpu::LCONST_0, 3, 3, cc_region);
        cc_region.close();
    };
    ::ckernel::sfpu::bf16_sfpi::sfpswap(0, 3, 1, 1);
    ::ckernel::sfpu::bf16_sfpi::sfpmad(13, 1, 14, 2, 0);
    if constexpr (Config::kDegree == 8) {
        reciprocal_complement_rung<Config, 0>();
        reciprocal_complement_rung<Config, 1>();
        reciprocal_complement_rung<Config, 2>();
        reciprocal_complement_rung<Config, 3>();
        reciprocal_complement_rung<Config, 4>();
        reciprocal_complement_rung<Config, 5>();
        reciprocal_complement_rung<Config, 6>();
    } else {
        reciprocal_complement_short_rung<Config, 0>();
    }
    {
        ::ckernel::sfpu::bf16_sfpi::Region cc_region;
        ::ckernel::sfpu::bf16_sfpi::sfpsetcc(0, 0, 0, 0, cc_region);
        ::ckernel::sfpu::bf16_sfpi::sfpmad(ckernel::p_sfpu::LCONST_neg1, 2, 4, 2, 0, cc_region);
        cc_region.close();
    };
    TTI_SFPLOADMACRO(3, 0, Advance, 0);
}
}  // namespace sfpi
