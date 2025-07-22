// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

enum class RelationalOp : uint8_t {
    EQ = 0,
    NE = 1,
    LT = 2,
    LE = 3,
    GT = 4,
    GE = 5,
};

template <bool APPROXIMATION_MODE, RelationalOp RELATIONAL_OP, int ITERATIONS = 8>
inline void calculate_sfpu_relational_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        vInt u = dst_reg[0];
        vInt v = dst_reg[dst_offset * dst_tile_size];
        vInt zero = 0;
        vInt res;
        // if constexpr (COMP_MODE == SfpuType::lt) {
        if constexpr (RELATIONAL_OP == RelationalOp::LT) {
            v_if(u < zero && v >= zero) { res = 1; }
            v_elseif(u >= zero && v < zero) { res = zero; }
            v_elseif(u < zero && v < zero) {
                res = u - v;
                v_if(res > zero) { res = 1; }
                v_else { res = zero; }
                v_endif;
            }
            v_else {
                res = u - v;
                v_if(res < zero) { res = 1; }
                v_else { res = zero; }
                v_endif;
            }
            v_endif;
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

}  //  namespace ckernel::sfpu
