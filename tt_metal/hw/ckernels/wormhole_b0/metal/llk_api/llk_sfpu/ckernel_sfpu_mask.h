// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask() {
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask)) { dst_reg[0] = 0.0f; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_int_mask() {
    const int mask_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt mask = dst_reg[mask_idx];
        v_if(mask == 0) { dst_reg[0] = 0.0f; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_posinf() {
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask)) { dst_reg[0] = std::numeric_limits<float>::infinity(); }
        v_endif;
        dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Mask<APPROX, FORMAT, FILL_POSINF, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode)
//   backs mask_tile (FORMAT Float16_b -> calculate_mask, Int32 -> calculate_int_mask), mask_posinf_tile
//   (FILL_POSINF -> calculate_mask_posinf) and mask_tile_init (shared SFPU init only).
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    DataFormat FORMAT,
    bool FILL_POSINF,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct Mask
    : SfpuUnaryOp<Mask<APPROXIMATION_MODE, FORMAT, FILL_POSINF, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static_assert(!(FILL_POSINF && FORMAT == DataFormat::Int32), "mask_posinf has no integer variant");

    static void kernel() {
        if constexpr (FILL_POSINF) {
            calculate_mask_posinf<APPROXIMATION_MODE, ITERATIONS>();
        } else if constexpr (FORMAT == DataFormat::Int32) {
            calculate_int_mask<APPROXIMATION_MODE, ITERATIONS>();
        } else {
            calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
