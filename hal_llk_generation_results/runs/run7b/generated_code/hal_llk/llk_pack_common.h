// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the pack-thread common functions (spec: llk_lib/llk_pack_common.h).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::pack
{

namespace cfg = hal::cfg;

// Point every packer Dest-offset register set at the Dest half currently owned by the packer.
inline void select_dest_half()
{
    const std::uint32_t offset = hal_llk::dest_half_base();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(offset),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S1>(offset),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S2>(offset),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S3>(offset));
}

inline void wait_packer_idle_for_config()
{
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
}

} // namespace hal_llk::pack

inline void _llk_packer_wait_for_math_done_()
{
    using namespace hal::sync;
    wait::semaphore<StallTarget::Pack, SemaphoreCondition::WhileZero, hal_llk::MATH_PACK_SEMAPHORE>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    using namespace hal::sync;
    // Every PACR of the section must be done before Dest is cleared and handed back to math.
    wait::stall<StallTarget::Math | StallTarget::Sync | StallTarget::Config, StallCondition::PackerIdle>();

    constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
    if constexpr (clear_dest)
    {
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else
        {
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = hal_llk::dest_half_index, .width = width});
        }
    }

    semaphore::get<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        hal_llk::dest_half_index ^= 1;
        hal_llk::pack::select_dest_half();
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    hal_llk::pack::wait_packer_idle_for_config();
    hal_llk::dest_half_index = 0;
    hal_llk::pack::select_dest_half();

    // Dest read position (channel 0) and L1 write position (channel 1) start at zero; the X range is owned by init.
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
    pack_sync_tile_dst_ptr = 0;
}
