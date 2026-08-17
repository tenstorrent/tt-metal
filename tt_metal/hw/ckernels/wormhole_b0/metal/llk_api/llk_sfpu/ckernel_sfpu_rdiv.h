// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, RoundingMode rounding_mode, int ITERATIONS>
inline void calculate_rdiv(const uint value) {
    sfpi::vFloat val = Converter::as_float(value);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat recip;
        if constexpr (APPROXIMATION_MODE) {
            recip = sfpu_reciprocal_iter<0>(in);
        } else {
            if constexpr (is_fp32_dest_acc_en) {
                recip = sfpu_reciprocal_iter<2>(in);
            } else {
                recip = sfpu_reciprocal_iter<1>(in);
                recip = sfpi::convert<sfpi::vFloat16b>(recip, sfpi::RoundMode::Nearest);
            }
        }
        sfpi::vFloat result = recip * val;

        if constexpr (rounding_mode == RoundingMode::Trunc) {
            result = _trunc_body_(result);
        } else if constexpr (rounding_mode == RoundingMode::Floor) {
            result = _floor_body_(result);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void rdiv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
