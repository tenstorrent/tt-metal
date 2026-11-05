// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"
#include "llk_sfpu/ckernel_sfpu_fill.h"

// MATH only. ZEROACC only sets Dest's zero flags; the words stay, so anything that reads Dest without
// going through those flags still sees topk's leftovers. Write real zeros once PACK has released Dest,
// so the next kernel starts clean however it reads.
inline __attribute__((noinline)) void topk_xl_scrub_dest()
{
    ckernel::tensix_sync();
    while (ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) > 0)
    {
    } // Wait for PACK to release every Dest section before touching it

    // Address Dest absolutely from row 0, whatever the sync mode.
    ckernel::reset_dest_offset_id();
    ckernel::math::set_dest_section_base<ckernel::StartZero>();

    constexpr std::uint32_t dest_tiles = ckernel::get_dest_max_tiles<DstSync::SyncFull, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();
    for (std::uint32_t tile = 0; tile < dest_tiles; tile++)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_int_<false, InstrModLoadStore::INT32, 8>, tile, VectorMode::RC, 0u /* raw zero word */);
    }
    // Callers that hand Dest off to PACK stall MATH+SFPU inside their t6_semaphore_post (topk_xl_split::start_pack);
    // the two end-of-math callers return into the kernel exit, which parks on pc_buf_base[0] and already waits for
    // the Tensix thread to be idle. A tensix_sync() here only costs a full pipeline drain. #62391.
}
