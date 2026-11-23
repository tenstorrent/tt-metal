// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the pack-side common LLK functions (dest sync with math).

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "llk_hal_common.h"

namespace hal_llk::pack
{

// Dest half the packer is currently reading (SyncHalf double buffering).
inline std::uint32_t dest_bank = 0;

// Point the packer at the dest half it reads next. Callers make sure the packer is idle.
inline void select_dest_bank()
{
    namespace cfg = hal::cfg;
    if (dest_bank == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, 0>(), cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, DEST_HALF_ROWS>(), cfg::set<cfg::DestTargetRegCfgPack::ZOffset, cfg::Sec::S0, 0>());
    }
}

} // namespace hal_llk::pack

/**
 * @brief Stall the packer until math has finished a dest section.
 */
inline void _llk_packer_wait_for_math_done_()
{
    hal::sync::wait::semaphore<hal::sync::StallTarget::Pack, hal::sync::SemaphoreCondition::WhileZero, hal_llk::SEM_MATH_PACK>();
}

/**
 * @brief Release the dest section back to math.
 */
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::Sync, hal::sync::StallCondition::PackerIdle>();
    }
    hal::sync::semaphore::get<hal_llk::SEM_MATH_PACK>();
}

/**
 * @brief Finish a dest section: wait for the packer, clear the packed half, release it to math, flip halves.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    using namespace hal_llk;
    constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;

    // Every following instruction (dest clear, semaphore, packer reconfiguration) waits for the packer.
    hal::sync::wait::stall<hal::sync::StallTarget::Math, hal::sync::StallCondition::PackerIdle>();
    if constexpr (clear_dest)
    {
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else if (pack::dest_bank == 0)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 0, .width = width}>();
        }
        else
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 1, .width = width}>();
        }
    }
    _llk_packer_set_math_semaphore_();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        pack::dest_bank = 1 - pack::dest_bank;
        pack::select_dest_bank();
    }
}

template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
    hal_llk::pack::select_dest_bank();
}

/**
 * @brief Initialize packer dest state at the start of a kernel.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    hal_llk::pack::dest_bank = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();

    // Reset the packer address counters (the X range is owned by the pack init).
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
