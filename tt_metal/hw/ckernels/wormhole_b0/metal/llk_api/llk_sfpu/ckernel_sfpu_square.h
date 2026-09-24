// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void square_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

    // Two independent rows so the scheduler can overlap their MUL->RNE chains
    // and hide the software-rounding latency introduced for bf16 dest.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = v0 * v0;
        sfpi::vFloat r1 = v1 * v1;
        if constexpr (!is_fp32_dest_acc_en) {
            r0 = float32_to_bf16_rne(r0);
            r1 = float32_to_bf16_rne(r1);
        }
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1] = r1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
