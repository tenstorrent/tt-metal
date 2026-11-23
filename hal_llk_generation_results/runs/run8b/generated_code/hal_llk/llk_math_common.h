// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL implementation of the math-thread configuration and Dest synchronization (Blackhole).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

using namespace ckernel;

namespace hal_llk
{

inline std::uint32_t math_dest_bank_base()
{
    return math_dest_bank ? DEST_HALF_ROWS : 0;
}

// Point the math Dest window at row `rows` (thread CFG, one SETC16).
inline void math_set_dest_offset(const std::uint32_t rows)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(rows);
}

} // namespace hal_llk

/**
 * @brief Configure the math thread's ALU control for the given source formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg = hal::cfg;

    const bool int8_math = hal_llk::is_int8_math_format(srca_data_format) || hal_llk::is_int8_math_format(srcb_data_format);
    // Dest holds 32-bit data in FP32 accumulation mode (Int32 for integer math), otherwise the math format.
    const std::uint32_t dest_format = is_fp32_dest_acc_en ? static_cast<std::uint32_t>(int8_math ? DataFormat::Int32 : DataFormat::Float32)
                                                          : hal_llk::format_code(srca_data_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg2::Dstacc, cfg::Sec::S0>(hal_llk::format_code(dest_format)),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));
    // FP compute on float / int8 operands flushes zero Src datums (zero-flag detection enabled).
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 0>();
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    namespace sync = hal::sync;
    sync::wait::semaphore<sync::StallTarget::Math, sync::SemaphoreCondition::WhileMaximum, hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Hand the finished Dest section to the packer; in SyncHalf mode switch to the other half.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace sync = hal::sync;

    // Every math write of the section must land before the packer may read it.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::post<hal_llk::MATH_PACK_SEM>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        hal_llk::math_dest_bank = 1 - hal_llk::math_dest_bank;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
        hal_llk::math_set_dest_offset(hal_llk::math_dest_bank_base());
    }
}

/**
 * @brief Initialize the math/pack Dest semaphore and reset the Dest section base.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace sync = hal::sync;

    // Packs of a previous section must be finished before math claims Dest.
    while (sync::semaphore::read<hal_llk::MATH_PACK_SEM>() > 0)
    {
    }
    constexpr std::uint32_t sections = (Dst == DstSync::SyncFull) ? 1 : 2;
    sync::semaphore::init<0, sections, hal_llk::MATH_PACK_SEM>();

    hal_llk::math_dest_bank = 0;
    hal_llk::math_set_dest_offset(hal_llk::math_dest_bank_base());
}
