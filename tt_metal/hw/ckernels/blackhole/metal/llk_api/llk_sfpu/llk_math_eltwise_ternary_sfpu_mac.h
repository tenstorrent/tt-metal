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
    // All callers use tile indices (0, 1, 2, 0), i.e. dst_reg[0], dst_reg[32], dst_reg[64]
    // (sfpi indexes dst_reg in rows; the row stride is 2, so tile n starts at row n * 32).
    //
    // The SFPLOAD/SFPSTORE instruction mod describes the DST register layout, which is
    // 32-bit only when fp32 dest accumulation is enabled - it is not a property of the
    // input tensor's data format.  Keying it off data_format would issue FP32-mode
    // accesses against a 16-bit DST whenever the inputs are fp32 but the output dtype
    // (which is what drives fp32_dest_acc_en) is bf16, and 16-bit accesses against a
    // 32-bit DST in the mirrored case.  DataLayout::FSrcB is mod 0, i.e. "whatever
    // format DST is configured with".
    //
    // sfpi defaults the loads to the arch's no-increment addr_mod (ADDR_MOD_7 here), which
    // is what we want; only the store overrides it, with the auto-advancing mod set above.
    constexpr sfpi::DataLayout dst_fmt = is_fp32_dest_acc_en ? sfpi::DataLayout::F32 : sfpi::DataLayout::FSrcB;

    lltt::record<lltt::NoExec>(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    sfpi::vFloat a = sfpi::dst_reg[0].mode<dst_fmt>();
    sfpi::vFloat b = sfpi::dst_reg[32].mode<dst_fmt>();
    sfpi::vFloat c = sfpi::dst_reg[64].mode<dst_fmt>();
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::dst_reg[0].mode<dst_fmt>(ADDR_MOD_6) = a * b + c;
    } else {
        // SFPMAD accumulates in fp32; an SFPSTORE into a 16-bit DST truncates the
        // mantissa, so round to nearest-even on the way down to bf16 instead.
        sfpi::dst_reg[0].mode<dst_fmt>(ADDR_MOD_6) =
            sfpi::convert<sfpi::vFloat16b>(a * b + c, sfpi::RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu
