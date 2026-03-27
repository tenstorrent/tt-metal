// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// mac: out = a * b + c, computed in FP32 accumulator via SFPMAD.
//
// ADDR_MOD_6 on SFPSTORE auto-advances the dest base register by 2 rows per
// replay, so the next replay's SFPLOADs read the next row group automatically.
// This avoids the explicit sfpi::dst_reg++ used in a plain for-loop, which
// only advances the write counter and not the read counter.
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
    // Each tile occupies 64 rows in the dest register file.
    const uint offset_in0 = dst_index_in0 * 64;
    const uint offset_in1 = dst_index_in1 * 64;
    const uint offset_in2 = dst_index_in2 * 64;
    const uint offset_out  = dst_index_out  * 64;

    if constexpr (is_fp32_dest_acc_en) {
        // FP32 dest: no stochastic rounding → 6-instruction replay sequence.
        lltt::record(0, 6);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset_in0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset_in1);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, offset_in2);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        // ADDR_MOD_6 (dest.incr=2, configured in mac_init) advances the base by
        // 2 rows after each SFPSTORE so that the next replay reads the next row group.
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_6, offset_out);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            lltt::replay(0, 6);
        }
    } else {
        // BF16 dest: stochastic rounding FP32→FP16B → 7-instruction replay sequence.
        lltt::record(0, 7);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset_in0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset_in1);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, offset_in2);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_6, offset_out);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            lltt::replay(0, 7);
        }
    }
}

}  // namespace ckernel::sfpu
