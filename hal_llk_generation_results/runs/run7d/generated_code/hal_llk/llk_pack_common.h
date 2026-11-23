// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the pack-thread Dest synchronization and Dest section selection.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/sync.h"
#include "hal_llk_common.h"
#include "llk_memory_checks.h"
#include "llk_defs.h"
#include "llk_math_common.h"

using namespace ckernel;

namespace hal_llk::pack_detail
{

namespace cfg = hal::cfg;

// Point all four packer Dest read interfaces at the current Dest section.
inline void select_pack_dest_section()
{
    const std::uint32_t base = dest_section_base();
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

// Zero the packer's Y/Z/W counters on both channels and the channel-0 X start.
inline void reset_packer_counters()
{
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .X<0>()
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
    using namespace hal::sync;
    wait::semaphore<StallTarget::Pack, SemaphoreCondition::WhileZero, hal_llk::MATH_PACK_SEMAPHORE>();
}

/**
 * @brief Finish a destination-register section: wait for pack, optionally clear dest, and release math.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    using namespace hal::sync;

    // The section may only be cleared once the packer has read all of it.
    wait::stall<StallTarget::Math, StallCondition::PackerIdle>();
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

    // Release the section to math once the clear has completed.
    wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    semaphore::get<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        ckernel::dest_offset_id = 1 - ckernel::dest_offset_id;
        wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
        hal_llk::pack_detail::select_pack_dest_section();
    }
}

/**
 * @brief Initialize the packer destination-offset registers for the selected sync mode.
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    using namespace hal::sync;
    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    hal_llk::pack_detail::select_pack_dest_section();
}

/**
 * @brief Initialize packer destination state at the start of a kernel.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    ckernel::dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_llk::pack_detail::reset_packer_counters();
    pack_sync_tile_dst_ptr = 0;
}
