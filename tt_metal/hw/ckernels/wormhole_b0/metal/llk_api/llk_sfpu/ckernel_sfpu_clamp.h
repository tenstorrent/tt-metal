// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

enum { Max = true, Min = false };  // Clamp Mode

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(min_val);
        calculate_unary_max_min_float_body<Max>();
        load_value_param_float(max_val);
        calculate_unary_max_min_float_body<Min>();
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint min_val, uint max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_int(min_val);
        calculate_unary_max_min_int32_body<Max>(min_val);
        load_value_param_int(max_val);
        calculate_unary_max_min_int32_body<Min>(max_val);
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Clamp<APPROX, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, min_val, max_val) -> calculate_clamp (float) / calculate_clamp_int32 (Int32)
//                                                          (clamp_tile, clamp_tile_int32)
//   init()                                              -> bare init      (clamp_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Clamp : SfpuUnaryOp<Clamp<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t min_val, uint32_t max_val) {
        if constexpr (FORMAT == DataFormat::Int32) {
            calculate_clamp_int32<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
        } else {
            calculate_clamp<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
        }
    }
};
}  // namespace ckernel::sfpu
