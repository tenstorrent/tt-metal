// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <typename V, typename T>
inline void calculate_logical_not_unary() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        V v = sfpi::dst_reg[0];
        v_if(v == 0) { sfpi::dst_reg[0] = T(1); }
        v_else { sfpi::dst_reg[0] = T(0); }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
