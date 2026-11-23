// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole math common LLK functions.

#pragma once

#include <cstdint>

#include "hal/math_counters.h"
#include "llk_hal_common.h"

/**
 * @brief Program the ALU accumulation mode for the given source formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;
    using hal_llk::fmt;

    const std::uint32_t int8_math_enabled = hal_llk::is_int8_math_format(srca_data_format) || hal_llk::is_int8_math_format(srcb_data_format);
    // MOVA2D/MOVB2D flush datums with a zero low byte unless the zero flag is disabled; only UInt16 needs that.
    const std::uint32_t disable_src_zero_flag =
        ((srca_data_format & 0x1F) == fmt(DataFormat::UInt16)) || ((srcb_data_format & 0x1F) == fmt(DataFormat::UInt16));

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math_enabled),
        cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(disable_src_zero_flag));
    sync::mutex::release<sync::Mutex::M0>();
}

/**
 * @brief Block math until the packer has released a Dest section.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    namespace sync = hal_llk::sync;
    sync::wait::semaphore<sync::StallTarget::Math, sync::SemaphoreCondition::WhileMaximum, hal_llk::SEM_MATH_PACK>();
}

/**
 * @brief Hand the finished Dest section to the packer and, in half-sync mode, move to the other half.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace sync = hal_llk::sync;

    // All Dest writes of this section must land before the packer is signalled.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::semaphore::post<hal_llk::SEM_MATH_PACK>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        ckernel::dest_offset_id  = 1 - ckernel::dest_offset_id;
    }
}

/**
 * @brief Initialize the math <-> pack Dest handshake.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace cfg  = hal_llk::cfg;
    namespace sync = hal_llk::sync;

    // Previous packs must drain before math claims Dest.
    while (sync::semaphore::read<hal_llk::SEM_MATH_PACK>() > 0)
    {
    }
    constexpr std::uint8_t num_sections = (Dst == DstSync::SyncFull) ? 1 : 2;
    sync::semaphore::init<0, num_sections, hal_llk::SEM_MATH_PACK>();

    ckernel::dest_offset_id = 0;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, 0>();
}
