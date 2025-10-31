// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_alt_complex_rotate90() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -sfpi::dst_reg[1];
        sfpi::dst_reg[1] = val;
        sfpi::dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
