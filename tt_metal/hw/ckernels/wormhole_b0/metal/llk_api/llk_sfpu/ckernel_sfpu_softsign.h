// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "llk_defs.h"

namespace ckernel::sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat tmp = sfpi::abs(v) + sfpi::vConst1;
        tmp = sfpu_reciprocal<(APPROX_MODE == ApproximationMode::Fast)>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE>
void init_softsign() {
    sfpu_reciprocal_init<(APPROX_MODE == ApproximationMode::Fast)>();
}

}  // namespace ckernel::sfpu
