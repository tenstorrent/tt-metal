// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) {
    // SFPU microcode
    sfpi::vFloat init = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if(a < 0.0f) { a = a * init; }
        v_endif;
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
