// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the math-thread configuration and math/pack Dest synchronization.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal_llk_common.h"
#include "llk_defs.h"

using namespace ckernel;

namespace hal_llk
{

// Semaphore counting Dest sections produced by math and not yet released by pack (ckernel::semaphore::MATH_PACK).
constexpr hal::sync::Semaphore MATH_PACK_SEMAPHORE = hal::sync::Semaphore::S1;

// First Dest row of the section selected by dest_offset_id.
inline std::uint32_t dest_section_base()
{
    return (ckernel::dest_offset_id != 0) ? DEST_REGISTER_HALF_SIZE : 0;
}

} // namespace hal_llk

namespace hal_llk::math_detail
{

// Point the math thread's Dest window at the current section.
inline void select_math_dest_section()
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_section_base());
}

} // namespace hal_llk::math_detail

/**
 * @brief Configure the math (FPU) thread's ALU control registers for the given source data formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;
    using namespace hal_llk;

    wait::stall<StallTarget::Config, StallCondition::MathIdle>();

    // Releasing a source bank also clears its data-valid flag (normal unpacker hand-back).
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    const bool int8_math = is_int8_math_format(srca_data_format) || is_int8_math_format(srcb_data_format);
    // UInt16 sources carry integer bits in the exponent position; keep their zero datums intact.
    const bool keep_src_zeros = is_format(srca_data_format, DataFormat::UInt16) || is_format(srcb_data_format, DataFormat::UInt16);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0),
        cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(keep_src_zeros ? 1 : 0));
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    using namespace hal::sync;
    // Pack releases a section by decrementing the semaphore; wait while all sections are still full.
    wait::semaphore<StallTarget::Math | StallTarget::Sfpu | StallTarget::Sync, SemaphoreCondition::WhileMaximum, hal_llk::MATH_PACK_SEMAPHORE>();
}

/**
 * @brief Signal completion of a destination section, flipping banks in half-sync mode.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    using namespace hal::sync;

    // Everything written to the section must land before pack is told about it.
    wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    semaphore::post<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        ckernel::dest_offset_id  = 1 - ckernel::dest_offset_id;
        wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
        hal_llk::math_detail::select_math_dest_section();
    }
}

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    using namespace hal::sync;

    // Packs of a previous kernel must have released every section before the semaphore is reseeded.
    while (semaphore::read<hal_llk::MATH_PACK_SEMAPHORE>() > 0)
    {
    }

    constexpr std::uint32_t sections = (Dst == DstSync::SyncFull) ? 1 : 2;
    semaphore::init<0, sections, hal_llk::MATH_PACK_SEMAPHORE>();

    ckernel::dest_offset_id = 0;
    wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    hal_llk::math_detail::select_math_dest_section();
}
