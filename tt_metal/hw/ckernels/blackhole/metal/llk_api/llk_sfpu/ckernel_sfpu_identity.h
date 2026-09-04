// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_sfpu_op.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_identity_uint() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Identity<APPROX, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode)
//   FORMAT == UInt32 -> calculate_identity_uint (vUInt copy); anything else -> calculate_identity (vFloat copy).
// Backs identity_tile / identity_tile_uint32 / identity_tile_init (shared SFPU init only).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Identity
    : SfpuUnaryOp<Identity<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        if constexpr (FORMAT == DataFormat::UInt32) {
            calculate_identity_uint<APPROXIMATION_MODE, ITERATIONS>();
        } else {
            calculate_identity<APPROXIMATION_MODE, ITERATIONS>();
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
