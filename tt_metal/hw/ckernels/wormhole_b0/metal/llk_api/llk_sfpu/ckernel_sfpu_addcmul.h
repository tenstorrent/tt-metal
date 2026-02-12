// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const uint dst_index_in0,  // input_a
    const uint dst_index_in1,  // input_b
    const uint dst_index_in2,  // input_c
    const uint dst_index_out,  // output
    const uint value) {        // scalar value to multiply with input_b
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    // addcmul = input_a + ((value * input_b) * input_c)
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, dst_index_in2 * dst_tile_size);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);
        TTI_SFPNOP;
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A,
                0,
                p_sfpu::LREG5,
                p_sfpu::LREG5,
                InstrModLoadStore::FP16A);
        }
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
