// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH) || defined(TRISC_PACK) || defined(LLK_TRISC_PACK)) && \
    !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_softsign_bf16.h"
#define TT_POLY_SOFTSIGN_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH)) && !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_softsign_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_abs_denominator<ttpoly_generated::SoftsignBf16Config, ITERATIONS>();
}
template <auto...>
inline void init_softsign_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_abs_denominator<ttpoly_generated::SoftsignBf16Config>();
}
#endif

}  // namespace ckernel::sfpu
