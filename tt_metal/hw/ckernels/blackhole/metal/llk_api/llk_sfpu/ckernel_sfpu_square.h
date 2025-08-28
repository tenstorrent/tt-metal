// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "llk_defs.h"
namespace ckernel::sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_square() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = in * in;

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
