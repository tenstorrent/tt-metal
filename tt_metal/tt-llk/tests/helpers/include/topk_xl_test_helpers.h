// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_pack_common.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

// Supply the Metal wrapper defaults; the tested call passes its sync mode explicitly.
constexpr bool DST_ACCUM_MODE   = true;
constexpr DstSync DST_SYNC_MODE = DstSync::SyncFull;
#include "experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h"

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

    // Exercise the production wrapper, including its tile-to-row address conversion.
    ckernel::llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values<K, Dst>(dst_index);

    if constexpr (sfpu::topk_xl_blaze_compat)
    {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    }
}

} // namespace ckernel::test
