// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void prelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) {
    // SFPU microcode
    vFloat init = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * init; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH) || defined(TRISC_PACK) || defined(LLK_TRISC_PACK)) && \
    !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_prelu_bf16.h"
#define TT_POLY_PRELU_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if (defined(TRISC_MATH) || defined(LLK_TRISC_MATH)) && !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_prelu_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_slope_max<ttpoly_generated::PreluBf16Config, ITERATIONS>();
}
inline void init_prelu_tt_poly_bf16() { ckernel::sfpu::ttpoly::init_slope_max<ttpoly_generated::PreluBf16Config>(); }
#endif

}  // namespace ckernel::sfpu
