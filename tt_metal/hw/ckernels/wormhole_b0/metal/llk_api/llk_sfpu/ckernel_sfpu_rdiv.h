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
                // Round to bf16 inside the helper, before its power-of-two scaling, as
                // calculate_reciprocal does: rounding the scaled value instead flushes
                // 1/(+/-2**126) to zero and, with the FP32 seed, misrounds inputs whose
                // reciprocal is near a bf16 rounding boundary. Needs the bf16-fitted
                // seed loaded by rdiv_init.
                recip = sfpu_reciprocal_iter<1 /*max_iter*/, true /*round_to_bf16*/>(in);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void rdiv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Load the bf16-fitted seed for the precise bf16 path of calculate_rdiv, matching recip_init.
    sfpu_reciprocal_init<APPROXIMATION_MODE, !APPROXIMATION_MODE && !is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
