// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_recip.h"
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
            recip = _sfpu_reciprocal_<0>(in);
        } else {
            if constexpr (is_fp32_dest_acc_en) {
                recip = _sfpu_reciprocal_<2>(in);
            } else {
                recip = _sfpu_reciprocal_<1>(in);
                recip = sfpi::float_to_fp16b(recip, sfpi::RoundMode::NearestEven);
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
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
