// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_max() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[32];
        v_if(a < b) { dst_reg[0] = b; }
        v_endif;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
