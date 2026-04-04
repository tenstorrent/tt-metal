// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// atanh(x) = 0.5 * ln((1 + x) / (1 - x))
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_atanh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat abs_inp = sfpi::abs(inp);
        v_if(abs_inp > sfpi::vConst1) { sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(abs_inp == sfpi::vConst1) {
            sfpi::vFloat inf = std::numeric_limits<float>::infinity();
            sfpi::dst_reg[0] = sfpi::setsgn(inf, inp);
        }
        v_else {
            sfpi::vFloat num = sfpi::vConst1 + inp;
            sfpi::vFloat den = sfpi::vConst1 - inp;
            sfpi::vFloat tmp = _sfpu_reciprocal_<APPROXIMATION_MODE ? 0 : 2>(den);
            tmp = sfpi::setsgn(tmp, den);
            if constexpr (is_fp32_dest_acc_en || APPROXIMATION_MODE) {
                den = tmp;
            } else {
                den = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(tmp, 0));
            }
            num = num * den;
            den = _calculate_log_body_no_init_(num);
            sfpi::dst_reg[0] = 0.5f * den;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void atanh_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
