// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en>
inline void calculate_square() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = v * v;
        if constexpr (!fp32_dest_acc_en) {
            v = sfpi::convert<sfpi::vFloat16b>(v, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
