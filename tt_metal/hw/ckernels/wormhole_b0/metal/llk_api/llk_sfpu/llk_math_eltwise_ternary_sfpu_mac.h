// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_mac.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_mac(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_mac<APPROXIMATE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_mac_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::mac>();
    // eltwise_ternary_sfpu_configure_addrmod only sets ADDR_MOD_6 (dest.incr=2)
    // for SfpuType::where.  mac's replay sequence uses ADDR_MOD_2 on SFPSTORE
    // (which maps to physical slot 6 after set_addr_mod_base() adds 4), so we
    // must configure it explicitly here.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }.set(ADDR_MOD_6);

    // Record the replay sequence once at init time with fixed dest offsets.
    // All callers use tile indices (0, 1, 2, 0) → offsets (0, 64, 128, 0).
    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;
    if constexpr (is_fp32_dest_acc_en) {
        lltt::record(0, 6);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, 0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, 64);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, 128);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_2, 0);
    } else {
        lltt::record(0, 7);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, 0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, 64);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, 128);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_2, 0);
    }
}

}  // namespace ckernel
