// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal/gpr_ops.h"
#include "llk_hal_common.h"

namespace hal_llk
{

// GPRs of the pack thread (same roles as ckernel::p_gpr_pack).
inline constexpr std::uint32_t PACK_GPR_DEST_OFFSET_LO = 4;  // 4 GPRs: offsets of dest section 0
inline constexpr std::uint32_t PACK_GPR_DEST_OFFSET_HI = 8;  // 4 GPRs: offsets of dest section 1
inline constexpr std::uint32_t PACK_GPR_OUTPUT_ADDR    = 12; // L1 address of the tile being packed

// Point the packer at the dest section this thread currently owns: one 128-bit WRCFG copies the
// four per-register-set offsets from the GPR quad prepared by _llk_init_packer_dest_offset_registers_.
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    namespace cfg = hal::cfg;
    if (Dst == DstSync::SyncHalf && ckernel::dest_offset_id != 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(
            hal::gpr<PACK_GPR_DEST_OFFSET_HI>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits128>(
            hal::gpr<PACK_GPR_DEST_OFFSET_LO>());
    }
}

} // namespace hal_llk

/**
 * @brief Stall the packer until math has finished a dest section.
 */
inline void _llk_packer_wait_for_math_done_()
{
    namespace sync = hal::sync;
    sync::wait::semaphore<sync::StallTarget::Pack | sync::StallTarget::Thcon, sync::SemaphoreCondition::WhileZero, hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Release the dest section to math (MATH_PACK semaphore get).
 */
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    namespace sync = hal::sync;
    if constexpr (WaitRes != p_stall::NONE)
    {
        sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();
    }
    sync::semaphore::get<hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Finish packing a dest section: wait for the packer, optionally clear the section, release it to math.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Math | sync::StallTarget::Sync, sync::StallCondition::PackerIdle>();

    if constexpr (clear_dest)
    {
        constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else if (ckernel::dest_offset_id != 0)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 1, .width = width}>();
        }
        else
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 0, .width = width}>();
        }
    }

    sync::semaphore::get<hal_llk::MATH_PACK_SEM>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        ckernel::dest_offset_id = 1 - ckernel::dest_offset_id;
        hal_llk::select_packer_dest_registers<Dst>();
    }
}

/**
 * @brief Load the dest-section offsets into the pack GPRs and select section 0.
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace sync = hal::sync;
    namespace gpr  = hal::gpr_ops;

    sync::wait::stall<sync::StallTarget::Thcon | sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    constexpr std::uint32_t hi = (Dst == DstSync::SyncHalf) ? hal_llk::DEST_HALF_ROWS : 0;
    gpr::set<0>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_LO + 0>());
    gpr::set<0>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_LO + 1>());
    gpr::set<0>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_LO + 2>());
    gpr::set<0>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_LO + 3>());
    gpr::set<hi>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_HI + 0>());
    gpr::set<hi>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_HI + 1>());
    gpr::set<hi>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_HI + 2>());
    gpr::set<hi>(hal::gpr<hal_llk::PACK_GPR_DEST_OFFSET_HI + 3>());

    hal_llk::select_packer_dest_registers<Dst>();
}

/**
 * @brief Initialise the packer dest state: section 0, dest offsets and packer address counters.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    ckernel::dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();

    // Packer counters: X start, Y, Z, W on both channels back to 0.
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
