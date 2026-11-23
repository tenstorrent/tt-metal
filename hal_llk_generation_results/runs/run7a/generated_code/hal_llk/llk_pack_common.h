// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole pack common LLK functions.

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "llk_hal_common.h"

namespace hal_llk::pack
{

using AdcClient  = hal::AddressCounterClient;
using AdcChannel = hal::AddressChannel;

// Point the packer at the Dest section it currently owns.
inline void select_packer_dest_section()
{
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(dest_section_base()), cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S0, 0>());
}

} // namespace hal_llk::pack

/**
 * @brief Block the packer until math has finished a Dest section.
 */
inline void _llk_packer_wait_for_math_done_()
{
    namespace sync = hal_llk::sync;
    sync::wait::semaphore<sync::StallTarget::Pack, sync::SemaphoreCondition::WhileZero, hal_llk::SEM_MATH_PACK>();
}

/**
 * @brief Release a Dest section back to math (optionally clearing it) after it has been packed.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace sync = hal_llk::sync;

    if constexpr (clear_dest)
    {
        // Packed data must be read out before it is cleared.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>();
        constexpr auto width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else
        {
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = ckernel::dest_offset_id, .width = width});
        }
    }

    // Release the section only after the pack (and clear) completed.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle | sync::StallCondition::MathIdle>();
    sync::semaphore::get<hal_llk::SEM_MATH_PACK>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        ckernel::dest_offset_id = 1 - ckernel::dest_offset_id;
        hal_llk::pack::select_packer_dest_section();
    }
}

template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    hal_llk::pack::select_packer_dest_section();
}

/**
 * @brief Reset the packer Dest section and address counters at kernel start.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    using hal_llk::pack::AdcChannel;
    using hal_llk::pack::AdcClient;

    ckernel::dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();

    // Reset the packer Y/Z/W address counters; the X (datum range) counters belong to the pack init.
    hal::address_counters.client<AdcClient::Packers>()
        .channel<AdcChannel::Channel0>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .channel<AdcChannel::Channel1>()
        .Y<0>()
        .Z<0>()
        .W<0>()
        .apply();
    pack_sync_tile_dst_ptr = 0;
}
