// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// float32: |x| clamped at 2**26; x * 0 + 1 in the residual turns inf and NaN inputs into NaN.
template <int ITERATIONS>
inline void calculate_softsign_fp32() {
    constexpr uint32_t kZero = p_sfpu::LCONST_0, kOne = p_sfpu::LCONST_1, kClamp = 13;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPSETSGN(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
        TTI_SFPSWAP(0, kClamp, p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG0, kZero, kOne, p_sfpu::LREG3, 0);
        TTI_SFPSWAP(0, kClamp, p_sfpu::LREG5, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG4, kZero, kOne, p_sfpu::LREG7, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG1, kOne, kOne, p_sfpu::LREG1, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPMAD(p_sfpu::LREG5, kOne, kOne, p_sfpu::LREG5, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPARECIP(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG1, 1);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG5, 1);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, kZero, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG5, kZero, p_sfpu::LREG7, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG6, 0);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG6, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, kZero, p_sfpu::LREG0, 2);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, kZero, p_sfpu::LREG4, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
    }
}

// bfloat16 and approximate: (1 + |x|) * 2**-64 keeps SFPARECIP in range for every finite x; undone after the multiply.
template <int ITERATIONS>
inline void calculate_softsign_bf16() {
    constexpr uint32_t kZero = p_sfpu::LCONST_0, kTwo = 12, kScale = 13;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPSETSGN(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
        TTI_SFPMAD(p_sfpu::LREG0, kZero, kTwo, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG1, kScale, kScale, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG4, kZero, kTwo, p_sfpu::LREG7, 0);
        TTI_SFPMAD(p_sfpu::LREG5, kScale, kScale, p_sfpu::LREG5, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPARECIP(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG1, 1);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG5, 1);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, kZero, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, kZero, p_sfpu::LREG6, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, kZero, p_sfpu::LREG0, 2);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, kZero, p_sfpu::LREG4, 2);
        TTI_SFPMAD(p_sfpu::LREG0, kScale, kZero, p_sfpu::LREG0, 2);
        TTI_SFPMAD(p_sfpu::LREG4, kScale, kZero, p_sfpu::LREG4, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
    }
}

template <int ITERATIONS>
inline void calculate_softsign_approx() {
    constexpr uint32_t kZero = p_sfpu::LCONST_0, kScale = 13;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPSETSGN(0, p_sfpu::LREG4, p_sfpu::LREG5, 1);
        TTI_SFPMAD(p_sfpu::LREG1, kScale, kScale, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG5, kScale, kScale, p_sfpu::LREG5, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPARECIP(0, p_sfpu::LREG5, p_sfpu::LREG6, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, kZero, p_sfpu::LREG0, 2);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, kZero, p_sfpu::LREG4, 2);
        TTI_SFPMAD(p_sfpu::LREG0, kScale, kZero, p_sfpu::LREG0, 2);
        TTI_SFPMAD(p_sfpu::LREG4, kScale, kZero, p_sfpu::LREG4, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 2);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_softsign() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_softsign_approx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        calculate_softsign_fp32<ITERATIONS>();
    } else {
        calculate_softsign_bf16<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_softsign() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 2.0f + 0x1.0p-10f;  // biased up so the bfloat16 store does not truncate below 1
    }
    sfpi::vConstFloatPrgm1 = (!APPROXIMATION_MODE && is_fp32_dest_acc_en) ? 0x1.0p26f : 0x1.0p-64f;
}

}  // namespace ckernel::sfpu
