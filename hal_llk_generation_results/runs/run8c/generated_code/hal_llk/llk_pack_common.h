// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the packer Dest synchronization helpers (Blackhole).

#pragma once

#include <cstdint>

#include "hal/dst.h"
#include "hal_llk_common.h"

namespace hal_llk
{

// Row of the current Dest section as seen by the packer.
inline std::uint32_t pack_dest_section_base()
{
    return (dest_offset_id != 0) ? DEST_HALF_BASE_ROW : 0;
}

// Point all packer Dest register sets at the current Dest section.
inline void select_packer_dest_section()
{
    namespace cfg = hal::cfg;
    using cfg::set;
    if (dest_offset_id == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, 0>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S1, 0>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S2, 0>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S3, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, DEST_HALF_BASE_ROW>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S1, DEST_HALF_BASE_ROW>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S2, DEST_HALF_BASE_ROW>(),
            set<cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S3, DEST_HALF_BASE_ROW>());
    }
}

} // namespace hal_llk

/**
 * @brief Stall the packer until math has filled a Dest section.
 */
inline void _llk_packer_wait_for_math_done_()
{
    using namespace hal::sync;
    wait::semaphore<StallTarget::All, SemaphoreCondition::WhileZero, hal_llk::MATH_PACK_SEMAPHORE>();
}

/**
 * @brief Release the Dest section to math (SEMGET on MATH_PACK), optionally after a stall.
 */
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    using namespace hal::sync;
    if constexpr (WaitRes != p_stall::NONE)
    {
        // NOT_IN_HAL: the template parameter is a raw p_stall mask (spec signature); it is forwarded as a HAL condition.
        wait::stall<StallTarget::Sync, static_cast<StallCondition>(WaitRes)>();
    }
    semaphore::get<hal_llk::MATH_PACK_SEMAPHORE>();
}

/**
 * @brief Finish a Dest section: wait for the packer, optionally clear the section, and release it to math.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    using namespace hal::sync;

    // The packer must have read the whole section before it is cleared or handed back.
    wait::stall<StallTarget::Math, StallCondition::PackerIdle>();
    if constexpr (clear_dest)
    {
        constexpr hal::dst::DestWidth width = is_fp32_dest_acc_en ? hal::dst::DestWidth::Bits32 : hal::dst::DestWidth::Bits16;
        if constexpr (Dst == DstSync::SyncFull)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::All, .width = width}>();
        }
        else if (dest_offset_id == 0)
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 0, .width = width}>();
        }
        else
        {
            hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::Half, .index = 1, .width = width}>();
        }
    }
    wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::PackerIdle>();
    semaphore::get<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        dest_offset_id = 1 - dest_offset_id;
        wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
        hal_llk::select_packer_dest_section();
    }
}

/**
 * @brief Select the packer Dest registers for the first section.
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    using namespace hal::sync;
    wait::stall<StallTarget::Config, StallCondition::PackerIdle>();
    hal_llk::select_packer_dest_section();
}

/**
 * @brief Initialize packer Dest state at the start of a kernel.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    dest_offset_id = 0;
    _llk_init_packer_dest_offset_registers_<Dst>();

    // Rewind the packer Y/Z/W address counters (X start/end are owned by the pack init).
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

/**
 * @brief Select which Dest tile the packer reads (channel-0 W counter).
 */
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    static_assert(!mutex_ADC, "mutex_ADC is not supported by the HAL implementation");
    // Note: an issued SETADC can overtake a preceding MOP's PACRs; _llk_pack_ therefore selects the tile inside its MOP.
    hal::runtime_address_counters.client<hal::AddressCounterClient::Packers>().channel<hal::AddressChannel::Channel0>().W(tile_index).apply();
}

namespace hal_llk
{
// Packer relu mode (low 4 bits) and threshold (high 16 bits) of a packed relu configuration word.
inline void program_pack_relu(const std::uint32_t relu_config)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0xF), set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xFFFF));
}
} // namespace hal_llk
