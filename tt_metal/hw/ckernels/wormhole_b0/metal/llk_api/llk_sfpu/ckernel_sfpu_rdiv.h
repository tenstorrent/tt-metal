// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

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
                recip = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(recip, 0));
            }
        }
        sfpi::dst_reg[0] = recip * val;
        if constexpr (rounding_mode == RoundingMode::Trunc) {
            sfpi::dst_reg[0] = _trunc_body_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        } else if constexpr (rounding_mode == RoundingMode::Floor) {
            sfpi::dst_reg[0] = _floor_body_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        }
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void rdiv_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
