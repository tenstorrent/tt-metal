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
#include "hal/math_counters.h"
#include "hal/sync.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

using namespace ckernel;

namespace hal_llk::math
{
namespace cfg = hal::cfg;

// Dest rows per half of the Dest register (double-buffered SyncHalf mode).
constexpr std::uint32_t DEST_HALF_ROWS = 512;
// Dest rows reserved per 32x32 tile slot.
constexpr std::uint32_t DEST_TILE_ROWS = 64;

inline std::uint32_t dest_section_base()
{
    return (dest_offset_id != 0) ? DEST_HALF_ROWS : 0;
}

// Point the math Dest window at the given row (absolute within Dest).
inline void set_dest_offset(const std::uint32_t row)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(row);
}

// Select the Dest slot of a 32x32 tile within the current section.
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    set_dest_offset(dest_section_base() + tile_index * DEST_TILE_ROWS);
}

} // namespace hal_llk::math

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg = hal::cfg;

    auto is_int8_math = [](const std::uint32_t format)
    {
        const std::uint32_t f = format & 0xF;
        return f == static_cast<std::uint32_t>(DataFormat::Int8) || f == static_cast<std::uint32_t>(DataFormat::Int32);
    };
    const bool int8_math = is_int8_math(srca_data_format) || is_int8_math(srcb_data_format);

    // Shared ALU control word: written with Tensix RMW so the unpack thread's fields are preserved.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));

    // Source zero-flag substitution stays enabled for FP compute; Dest zero flags are honoured.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 0>(), cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_dst, cfg::Sec::S0, 0>());
}

template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    namespace sync = hal::sync;
    // Math may write Dest only while the packer has not yet been handed every available section.
    sync::wait::semaphore<sync::StallTarget::Math | sync::StallTarget::Sfpu, sync::SemaphoreCondition::WhileMaximum, sync::Semaphore::S1>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace sync = hal::sync;

    // Hand the finished section to the packer once math has drained.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::semaphore::post<sync::Semaphore::S1>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_offset_id           = 1 - dest_offset_id;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
        hal_llk::math::set_dest_offset(hal_llk::math::dest_section_base());
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace sync = hal::sync;

    // Wait until the packer has released every section from a previous kernel.
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

    dest_offset_id = 0;
    hal_llk::math::set_dest_offset(0);
}
