// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

bool bf16_dest_abs();
template <int ITERATIONS>
void calculate_abs_bf16();
void init_abs_bf16();
// Whether BF16 DEST runs the generated abs kernel as one call over the whole tile.
inline constexpr bool abs_bf16_whole_tile = true;
// Sets up the generated BF16 abs kernel for the instance it serves.
template <bool bf16_kernel>
inline void abs_bf16_tile_init() {
    if constexpr (bf16_kernel) {
        init_abs_bf16();
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = true>
inline void calculate_abs() {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_abs()) {
            calculate_abs_bf16<ITERATIONS>();
            return;
        }
    }
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, InstrModLoadStore::INT32, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::INT32, 3, 0);
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel

#include "ckernel_sfpu_abs_bf16.h"
