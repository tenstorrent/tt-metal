// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_mac(
    const uint dst_index_in0,  // input a
    const uint dst_index_in1,  // input b
    const uint dst_index_in2,  // input c
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_mac(). Supported data formats are: Float32, Float16_b.");

    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    // mac: out = a * b + c
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, dst_index_in2 * dst_tile_size);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
