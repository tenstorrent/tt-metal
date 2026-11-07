// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu {

inline void elu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    expm1_cw_init_prgm_consts();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    static_assert(ITERATIONS % 2 == 0, "calculate_elu processes rows in pairs");
    sfpi::vFloat alpha = Converter::as_float(slope);
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat ra = sfpi::dst_reg[0];
        sfpi::vFloat rb = sfpi::dst_reg[1];
        expm1_cw_clamped_pair(ra, rb);
        ra = alpha * ra;
        rb = alpha * rb;
        {
            sfpi::vFloat x = sfpi::dst_reg[0];
            v_if(x >= 0.0f) { ra = x; }
            v_endif;
        }
        {
            sfpi::vFloat x = sfpi::dst_reg[1];
            v_if(x >= 0.0f) { rb = x; }
            v_endif;
        }
        if constexpr (!is_fp32_dest_acc_en) {
            ra = sfpi::convert<sfpi::vFloat16b>(ra, sfpi::RoundMode::Nearest);
            rb = sfpi::convert<sfpi::vFloat16b>(rb, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = ra;
        sfpi::dst_reg[1] = rb;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
