// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_logical_not_unary() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v == 0) { sfpi::dst_reg[0] = 1.0f; }
        v_else { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_logical_not_unary_int32() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vInt v = sfpi::dst_reg[0];
        v_if(v == 0) { sfpi::dst_reg[0] = 1; }
        v_else { sfpi::dst_reg[0] = 0; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
