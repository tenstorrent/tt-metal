// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_mac.h"

namespace ckernel::sfpu {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format>
inline void mac_init() {
    // eltwise_ternary_sfpu_configure_addrmod only sets ADDR_MOD_6 (dest.incr=2)
    // for SfpuType::where.  mac's replay sequence uses ADDR_MOD_6 on SFPSTORE
    // (physical slot 6, direct on Blackhole since there is no addr_mod_base
    // offset), so we must configure it explicitly here.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }.set(ADDR_MOD_6);

    // Record the replay sequence once at init time with fixed dest offsets.
    // All callers use tile indices (0, 1, 2, 0) → offsets (0, 64, 128, 0).
    // The SFPLOAD/SFPSTORE instruction mod describes the DST register layout, which is
    // 32-bit only when fp32 dest accumulation is enabled - it is not a property of the
    // input tensor's data format.  Keying it off data_format would issue FP32-mode
    // accesses against a 16-bit DST whenever the inputs are fp32 but the output dtype
    // (which is what drives fp32_dest_acc_en) is bf16, and 16-bit accesses against a
    // 32-bit DST in the mirrored case.
    constexpr InstrModLoadStore mod0 = is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;
    if constexpr (is_fp32_dest_acc_en) {
        lltt::record(0, 6);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, 0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, 64);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, 128);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_6, 0);
    } else {
        lltt::record(0, 7);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, 0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, 64);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, 128);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TT_SFPSTORE(p_sfpu::LREG3, mod0, ADDR_MOD_6, 0);
    }
}

}  // namespace ckernel::sfpu
