// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the Blackhole math common LLK functions.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_defs.h"

using namespace ckernel;

namespace hal_llk::math
{

/// Semaphore math and pack use to hand over Dest sections (ckernel::semaphore::MATH_PACK).
inline constexpr hal::sync::Semaphore MathPackSemaphore = hal::sync::Semaphore::S1;

/// Source formats given to _llk_math_hw_configure_ (masked to the 4-bit hardware code).
inline std::uint32_t srca_format = 0;
inline std::uint32_t srcb_format = 0;

/// Dest rows occupied by one tile index (four 16-row faces).
inline constexpr std::uint32_t DestRowsPerTile = 64;

/// First Dest row of the half section math currently writes.
inline std::uint32_t dest_section_base()
{
    return (ckernel::dest_offset_id == 0) ? 0 : DEST_REGISTER_HALF_SIZE;
}

/// Point math Dest addressing at the section base.
inline void set_dest_offset_to_section_base()
{
    using namespace hal;
    if (ckernel::dest_offset_id == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, 0>();
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, DEST_REGISTER_HALF_SIZE>();
    }
}

inline constexpr bool is_int8_or_int32(const std::uint32_t format)
{
    const std::uint32_t masked = format & DATA_FORMAT_CONFIG_MASK;
    return masked == (static_cast<std::uint32_t>(DataFormat::Int8) & DATA_FORMAT_CONFIG_MASK) ||
           masked == (static_cast<std::uint32_t>(DataFormat::Int32) & DATA_FORMAT_CONFIG_MASK);
}

} // namespace hal_llk::math

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    using namespace hal;
    const std::uint32_t int8_math = (hal_llk::math::is_int8_or_int32(srca_data_format) || hal_llk::math::is_int8_or_int32(srcb_data_format)) ? 1 : 0;
    hal_llk::math::srca_format    = srca_data_format & DATA_FORMAT_CONFIG_MASK;
    hal_llk::math::srcb_format    = srcb_data_format & DATA_FORMAT_CONFIG_MASK;

    // Accumulation mode must not change under an in-flight math instruction.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>());
}

template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    using namespace hal::sync;
    // Math may write Dest while the packer holds fewer sections than the semaphore maximum.
    wait::semaphore<StallTarget::Math, SemaphoreCondition::WhileMaximum, hal_llk::math::MathPackSemaphore>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    using namespace hal::sync;
    // Hand the finished section to the packer once every math write to it has completed.
    wait::stall<StallTarget::Sync, StallCondition::MathIdle>();
    semaphore::post<hal_llk::math::MathPackSemaphore>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        ckernel::dest_offset_id  = 1 - ckernel::dest_offset_id;
        wait::stall<StallTarget::Config, StallCondition::MathIdle>();
        hal_llk::math::set_dest_offset_to_section_base();
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    using namespace hal::sync;
    // Let the packer drain every section from a previous kernel before claiming Dest.
    while (semaphore::read<hal_llk::math::MathPackSemaphore>() > 0)
    {
    }
    if constexpr (Dst == DstSync::SyncFull)
    {
        semaphore::init<0, 1, hal_llk::math::MathPackSemaphore>();
    }
    else
    {
        semaphore::init<0, 2, hal_llk::math::MathPackSemaphore>();
    }
    ckernel::dest_offset_id = 0;
    hal_llk::math::set_dest_offset_to_section_base();
}
