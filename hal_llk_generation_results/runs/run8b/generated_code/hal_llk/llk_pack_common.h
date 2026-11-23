// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of packer Dest synchronization (Blackhole).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

using namespace ckernel;

namespace hal_llk
{

// Point the packer at the Dest half it reads next (the offsets were staged in GPRs by dest init).
inline void pack_select_dest_bank()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    if (pack_dest_bank == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(
            hal::gpr<GPR_PACK_DEST_OFFSET_LO>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(
            hal::gpr<GPR_PACK_DEST_OFFSET_HI>());
    }
}

// Reset the packer Y/Z/W address counters. The X counters (datum range) are owned by the pack init.
inline void packer_reset_address_counters()
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

} // namespace hal_llk

/**
 * @brief Stall the packer until the math thread has produced a Dest section.
 */
inline void _llk_packer_wait_for_math_done_()
{
    namespace sync = hal::sync;
    while (sync::semaphore::read<hal_llk::MATH_PACK_SEM>() == 0) // DBG
    {
    }
    sync::wait::semaphore<sync::StallTarget::Tdma, sync::SemaphoreCondition::WhileZero, hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Tell math it may write the Dest section again.
 */
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    hal::sync::semaphore::get<hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Finish a Dest section: wait for the packs, clear the section, release math, switch halves.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace sync = hal::sync;

    // Dest may only be cleared once the packer has read all of it.
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>();
    if constexpr (clear_dest)
    {
        constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else if (hal_llk::pack_dest_bank == 0)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 0, .width = width}>();
        }
        else
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 1, .width = width}>();
        }
    }
    // Release the section only after the clear has completed.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::get<hal_llk::MATH_PACK_SEM>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        hal_llk::pack_dest_bank = 1 - hal_llk::pack_dest_bank;
        hal_llk::pack_select_dest_bank();
    }
}

/**
 * @brief Stage the Dest-half offsets in GPRs and select the first half.
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace gpr_ops = hal::gpr_ops;
    using namespace hal_llk;
    // Every Dest-read interface uses the same section base.
    gpr_ops::set<0>(hal::gpr<GPR_PACK_DEST_OFFSET_LO + 0>());
    gpr_ops::set<0>(hal::gpr<GPR_PACK_DEST_OFFSET_LO + 1>());
    gpr_ops::set<0>(hal::gpr<GPR_PACK_DEST_OFFSET_LO + 2>());
    gpr_ops::set<0>(hal::gpr<GPR_PACK_DEST_OFFSET_LO + 3>());
    constexpr std::uint32_t hi = (Dst == DstSync::SyncHalf) ? DEST_HALF_ROWS : 0;
    gpr_ops::set<hi>(hal::gpr<GPR_PACK_DEST_OFFSET_HI + 0>());
    gpr_ops::set<hi>(hal::gpr<GPR_PACK_DEST_OFFSET_HI + 1>());
    gpr_ops::set<hi>(hal::gpr<GPR_PACK_DEST_OFFSET_HI + 2>());
    gpr_ops::set<hi>(hal::gpr<GPR_PACK_DEST_OFFSET_HI + 3>());
    pack_dest_bank = 0;
    pack_select_dest_bank();
}

/**
 * @brief Initialize packer Dest state at the start of a kernel.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_llk::packer_reset_address_counters();
}

/**
 * @brief Select the Dest tile the packer reads (packer channel-0 W counter).
 */
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>()
        .channel<hal::AddressChannel::Channel0>()
        .Z(0)
        .W(tile_index)
        .channel<hal::AddressChannel::Channel1>()
        .Z(0)
        .W(tile_index)
        .apply();
}
