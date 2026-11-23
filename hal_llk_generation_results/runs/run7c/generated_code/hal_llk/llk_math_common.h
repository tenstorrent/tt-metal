// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the math-side common LLK functions (Blackhole).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_hal_common.h"

namespace hal_llk::math_detail
{

// Source formats seen by the last _llk_math_hw_configure_; they decide the Src zero-flag policy of compute inits.
inline std::uint32_t srca_format = 0;
inline std::uint32_t srcb_format = 0;

// First DEST row of the section the math thread currently writes.
inline std::uint32_t dest_section_base()
{
    return (ckernel::dest_offset_id != 0) ? DEST_HALF_ROWS : 0;
}

// Point the math thread's DEST window at `row` (absolute DEST row of the selected tile).
inline void set_dest_window(const std::uint32_t row)
{
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0>(row);
}

} // namespace hal_llk::math_detail

/**
 * @brief Configure the math (FPU) thread's ALU control registers for the given source data formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    using namespace hal_llk;

    math_detail::srca_format = srca_data_format;
    math_detail::srcb_format = srcb_data_format;

    // Consumed source banks are returned to the unpackers (data-valid cleared) on release.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    const std::uint32_t int8_math = is_int8_or_int32(srca_data_format) || is_int8_or_int32(srcb_data_format);
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
    sync::mutex::acquire<sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math));
    sync::mutex::release<sync::Mutex::M0>();
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    namespace sync = hal::sync;
    // The packer still owns every section: wait while MATH_PACK is at its maximum.
    sync::wait::semaphore<sync::StallTarget::Math | sync::StallTarget::Sync, sync::SemaphoreCondition::WhileMaximum, sync::Semaphore::S1>();
}

/**
 * @brief Signal completion of a destination section, flipping DEST halves in half-sync mode.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace sync = hal::sync;
    // Every math write of the section must land before the packer is told it may read it.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::post<sync::Semaphore::S1>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        ckernel::dest_offset_id  = 1 - ckernel::dest_offset_id;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
        hal_llk::math_detail::set_dest_window(hal_llk::math_detail::dest_section_base());
    }
}

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace sync = hal::sync;

    // Packs of an earlier section must drain before math claims every DEST section.
    while (sync::semaphore::read<sync::Semaphore::S1>() > 0)
    {
    }

    if constexpr (Dst == DstSync::SyncFull)
    {
        sync::semaphore::init<0 /* Initial */, 1 /* Maximum */, sync::Semaphore::S1>();
    }
    else
    {
        sync::semaphore::init<0 /* Initial */, 2 /* Maximum */, sync::Semaphore::S1>();
    }

    ckernel::dest_offset_id = 0;
    hal_llk::math_detail::set_dest_window(0);
}
