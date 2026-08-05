// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

inline void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out;
        if constexpr (DST_ACCUM_MODE) {
            out = ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
        } else {
            out = ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg += 2;
    }
}

template <uint16_t scale_bf16>
inline void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true, DST_ACCUM_MODE>(val, scale_bf16);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
