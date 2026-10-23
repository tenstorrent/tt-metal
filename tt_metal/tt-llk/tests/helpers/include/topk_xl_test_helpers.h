// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_pack_common.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

namespace ckernel::test
{

// Test-only PACK adapter. Call after waiting for MATH to release a SyncFull
// section; MATH must stay quiescent until PACK releases it. The compatibility
// LLK leaves the SFPU drains and ADDR_MOD_7 setup to its caller, so this adapter
// supplies them explicitly. The default LLK owns its drains and is initialized
// once by the driver before the row loop.
template <std::uint32_t K, DstSync Dst>
inline void topk_xl_pack_remove_msb_values(std::uint32_t dst_index)
{
    if constexpr (sfpu::topk_xl_blaze_compat)
    {
        // Re-establish PACK's address modifiers after the row's MATH init.
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::WAIT_SFPU);
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
        sfpu::_topk_xl_remove_msb_values_init_();
    }

    // Match the Metal wrapper's destination setup before invoking the raw LLK.
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH | p_stall::PACK);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    sfpu::_topk_xl_remove_msb_values_<K, Dst>();

    if constexpr (sfpu::topk_xl_blaze_compat)
    {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    }
}

} // namespace ckernel::test
