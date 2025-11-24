// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    // hardmish(x) = x * (x + 2.8).clamp(0.0, 5.0) / 5
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0] + 2.8f;

        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;

        v_if(a > 5.0f) { a = 5.0f; }
        v_endif;

        sfpi::dst_reg[0] = sfpi::dst_reg[0] * a * 0.2f;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
