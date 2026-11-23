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
namespace cfg = hal::cfg;

// Dest rows per half of the Dest register (double-buffered SyncHalf mode).
constexpr std::uint32_t DEST_HALF_ROWS = 512;

inline std::uint32_t dest_section_base()
{
    return (dest_offset_id != 0) ? DEST_HALF_ROWS : 0;
}

// Point the packer Dest window (all four packer register sets) at the active section.
inline void select_packer_dest_registers()
{
    const std::uint32_t base = dest_section_base();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(base),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S1>(base),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S2>(base),
        cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S3>(base));
}

// Datum range of one PACR: a single row of FACE_C_DIM datums. Observed on hardware: in the X-only
// (SETADCXX) form of the builder the range end is given as Channel0 X and the start as Channel1 X.
inline void set_packer_x_range()
{
    hal::address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .X<FACE_C_DIM - 1>()
        .channel<hal::AddressChannel::Channel1>()
        .X<0>()
        .apply();
}

// Reset the packer Y/Z/W address counters and program the per-PACR datum range.
inline void packer_addr_counter_init()
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
    set_packer_x_range();
}

} // namespace hal_llk::pack

inline void _llk_packer_wait_for_math_done_()
{
    namespace sync = hal::sync;
    // PACR issue goes through the TDMA queue, so that is what must be held back.
    sync::wait::semaphore<sync::StallTarget::Tdma | sync::StallTarget::Pack, sync::SemaphoreCondition::WhileZero, sync::Semaphore::S1>();
}

template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    namespace sync = hal::sync;
    if constexpr (WaitRes != p_stall::NONE)
    {
        sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();
    }
    sync::semaphore::get<sync::Semaphore::S1>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace sync = hal::sync;

    // Every pack of the section must have read Dest before it is cleared and handed back.
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
            hal::dst::run(hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = dest_offset_id % 2, .width = width});
        }
    }

    // Release the section to math once the clear has executed.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::get<sync::Semaphore::S1>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        dest_offset_id = 1 - dest_offset_id;
        hal_llk::pack::select_packer_dest_registers();
    }
}

template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    hal_llk::pack::select_packer_dest_registers();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_llk::pack::packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    static_assert(!mutex_ADC, "mutex_ADC is not supported by this implementation");
    // W selects the 32x32 tile slot of Dest.
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();
}

TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(static_cast<std::uint32_t>(relu_config.get_mode())),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(relu_config.get_threshold()));
}

inline void _llk_pack_reconfig_l1_acc_(const std::uint32_t enable)
{
    namespace cfg = hal::cfg;
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Pack_L1_Acc, cfg::Sec::S0>(enable ? 1 : 0);
}
