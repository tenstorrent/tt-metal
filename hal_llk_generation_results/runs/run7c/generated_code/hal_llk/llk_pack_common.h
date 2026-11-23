// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the pack-side common LLK functions (Blackhole).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_hal_common.h"

namespace hal_llk::pack_detail
{

// Point all four packer Dest-read register sets at the DEST section selected by dest_offset_id.
inline void select_packer_dest_section()
{
    namespace cfg            = hal::cfg;
    const std::uint32_t base = (ckernel::dest_offset_id != 0) ? DEST_HALF_ROWS : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(base),
        cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S0, 0>(),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S1>(base),
        cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S1, 0>(),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S2>(base),
        cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S2, 0>(),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S3>(base),
        cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S3, 0>());
}

// Restart the packer row/face/tile positions. X is left alone: it holds the per-PACR datum range owned by the
// pack init.
inline void reset_packer_address_counters()
{
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<hal::AddressChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();
}

} // namespace hal_llk::pack_detail

/**
 * @brief Stall the packer until the math thread has produced data to pack.
 */
inline void _llk_packer_wait_for_math_done_()
{
    namespace sync = hal::sync;
    sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>();
}

/**
 * @brief Signal the MATH_PACK semaphore (decrement) to release the math thread.
 */
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    static_assert(WaitRes == p_stall::NONE, "only the unstalled form is implemented");
    hal::sync::semaphore::get<hal::sync::Semaphore::S1>();
}

/**
 * @brief Initialize the packer destination-offset registers for the chosen sync mode.
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    hal_llk::pack_detail::select_packer_dest_section();
}

/**
 * @brief Initialize packer destination state at the start of a kernel.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    ckernel::dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_llk::pack_detail::reset_packer_address_counters();
    pack_sync_tile_dst_ptr = 0;
}

/**
 * @brief Finish a destination-register section: wait for pack, optionally clear dest, and release math.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace sync = hal::sync;

    // The section must be fully read out before it is cleared and handed back.
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>();
    if constexpr (clear_dest)
    {
        constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else
        {
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = ckernel::dest_offset_id % 2, .width = width});
        }
    }

    // Math may write the section again once the clear has executed.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::get<sync::Semaphore::S1>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        ckernel::dest_offset_id = 1 - ckernel::dest_offset_id;
        _llk_init_packer_dest_offset_registers_<Dst>();
    }
}
