// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    // lerp: out = input + weight * (end - input)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 + in2 * (in1 - in0);
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
