// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

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

}  // namespace ckernel::sfpu
