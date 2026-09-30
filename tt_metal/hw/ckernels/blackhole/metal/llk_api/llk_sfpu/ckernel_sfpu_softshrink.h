// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

inline void softshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH) || defined(TRISC_PACK) || defined(LLK_TRISC_PACK)) && \
    !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_softshrink_bf16.h"
#define TT_POLY_SOFTSHRINK_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH)) && !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_softshrink_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_simple_forward<ttpoly_generated::SoftshrinkBf16Config, ITERATIONS>();
}
inline void init_softshrink_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_simple_forward<ttpoly_generated::SoftshrinkBf16Config>();
}
#endif

}  // namespace ckernel::sfpu
