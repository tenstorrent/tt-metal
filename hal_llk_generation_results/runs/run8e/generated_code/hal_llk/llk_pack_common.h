// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole pack common LLK functions.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/address_counters.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

using namespace ckernel;

namespace hal_llk::pack
{

/// Semaphore math and pack use to hand over Dest sections (ckernel::semaphore::MATH_PACK).
inline constexpr hal::sync::Semaphore MathPackSemaphore = hal::sync::Semaphore::S1;

/// Point all four packer Dest-read interfaces at the Dest half the packer currently drains.
inline void select_dest_section()
{
    using namespace hal::cfg;
    if (ckernel::dest_offset_id == 0)
    {
        write<Access::TensixCfgUnit>(
            set<DestTargetRegCfgPack::Offset, Sec::S0, 0>(),
            set<DestTargetRegCfgPack::Offset, Sec::S1, 0>(),
            set<DestTargetRegCfgPack::Offset, Sec::S2, 0>(),
            set<DestTargetRegCfgPack::Offset, Sec::S3, 0>());
    }
    else
    {
        write<Access::TensixCfgUnit>(
            set<DestTargetRegCfgPack::Offset, Sec::S0, DEST_REGISTER_HALF_SIZE>(),
            set<DestTargetRegCfgPack::Offset, Sec::S1, DEST_REGISTER_HALF_SIZE>(),
            set<DestTargetRegCfgPack::Offset, Sec::S2, DEST_REGISTER_HALF_SIZE>(),
            set<DestTargetRegCfgPack::Offset, Sec::S3, DEST_REGISTER_HALF_SIZE>());
    }
}

} // namespace hal_llk::pack

inline void _llk_packer_wait_for_math_done_()
{
    using namespace hal::sync;
    wait::semaphore<StallTarget::Pack, SemaphoreCondition::WhileZero, hal_llk::pack::MathPackSemaphore>();
}

template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    static_assert(WaitRes == p_stall::NONE, "HAL pack implements the unstalled semaphore release only");
    hal::sync::semaphore::get<hal_llk::pack::MathPackSemaphore>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    using namespace hal;

    if constexpr (clear_dest)
    {
        // Zero the drained section only after the packer has read all of it.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>();
        constexpr dst::DestWidth width = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            dst::run<dst::Zero {.scope = dst::ZeroScope::All, .width = width}>();
        }
        else if (ckernel::dest_offset_id == 0)
        {
            dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .index = 0, .width = width}>();
        }
        else
        {
            dst::run<dst::Zero {.scope = dst::ZeroScope::Half, .index = 1, .width = width}>();
        }
    }
    else
    {
        sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();
    }

    // Return the section to math.
    _llk_packer_set_math_semaphore_();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        ckernel::dest_offset_id = 1 - ckernel::dest_offset_id;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
        hal_llk::pack::select_dest_section();
    }
}

template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
    hal_llk::pack::select_dest_section();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    ckernel::dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();

    // Rewind every packer address counter.
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
    pack_sync_tile_dst_ptr = 0;
}

template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    static_assert(!mutex_ADC, "HAL pack does not share the pack ADCs with other threads");
    // The Dest W counter selects the tile; the W stride is one Dest tile.
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();
}
