// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the math-thread common functions (spec: llk_lib/llk_math_common.h).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::math
{

namespace cfg = hal::cfg;

// Point the FPU at a Dest row base (Dest half base + tile offset).
inline void set_dest_offset(const std::uint32_t dest_row)
{
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row);
}

} // namespace hal_llk::math

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    const bool int8_math = hal_llk::is_int8_or_int32(srca_data_format) || hal_llk::is_int8_or_int32(srcb_data_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1 : 0));
}

template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    using namespace hal::sync;
    // Block FPU/SFPU work while the packer still owns every Dest section math may write.
    wait::semaphore<StallTarget::Math | StallTarget::Sfpu, SemaphoreCondition::WhileMaximum, hal_llk::MATH_PACK_SEMAPHORE>();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    using namespace hal::sync;
    // Publish the section only after every FPU/SFPU write to it has landed.
    wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    semaphore::post<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        hal_llk::dest_half_index ^= 1;
        wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
        hal_llk::math::set_dest_offset(hal_llk::dest_half_base());
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    using namespace hal::sync;
    // Wait for packs of a previous kernel to release Dest before claiming it.
    while (semaphore::read<hal_llk::MATH_PACK_SEMAPHORE>() > 0)
    {
    }
    constexpr std::uint8_t max_sections = (Dst == DstSync::SyncFull) ? 1 : 2;
    semaphore::init<0 /*Initial*/, max_sections /*Maximum*/, hal_llk::MATH_PACK_SEMAPHORE>();

    hal_llk::dest_half_index = 0;
    hal_llk::math::set_dest_offset(hal_llk::dest_half_base());
}
