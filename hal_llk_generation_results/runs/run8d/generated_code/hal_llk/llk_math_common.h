// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the math-side common LLK functions (dest sync, ALU configuration).

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk::math
{

// Dest half the math thread is currently writing (SyncHalf double buffering).
inline std::uint32_t dest_bank = 0;

inline std::uint32_t dest_bank_base()
{
    return dest_bank ? DEST_HALF_ROWS : 0;
}

} // namespace hal_llk::math

/**
 * @brief Configure the math thread's ALU controls for the given source formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    using namespace hal_llk;
    namespace cfg = hal::cfg;

    // Math releases consumed source banks back to the unpackers (normal data-valid handoff).
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    // Source formats are inferred from the unpacker on Blackhole: keep the implied formats enabled.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcA], cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DisableImpliedFmt[cfg::SrcB], cfg::Sec::S0, 0>();

    const bool int8_math = is_int8_or_int32(srca_data_format) || is_int8_or_int32(srcb_data_format);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math ? 1u : 0u),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluAccCtrl::Zero_Flag_disabled_dst, cfg::Sec::S0, 0>());
}

/**
 * @brief Block math until the packer has released a dest section.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    hal::sync::wait::semaphore<
        hal::sync::StallTarget::Math | hal::sync::StallTarget::Sfpu,
        hal::sync::SemaphoreCondition::WhileMaximum,
        hal_llk::SEM_MATH_PACK>();
}

/**
 * @brief Hand the finished dest section to the packer (and flip halves in SyncHalf mode).
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    using namespace hal_llk;
    // The section is complete only once every math/SFPU write to dest has retired.
    hal::sync::wait::stall<hal::sync::StallTarget::Sync, hal::sync::StallCondition::MathIdle | hal::sync::StallCondition::SfpuIdle>();
    hal::sync::semaphore::post<SEM_MATH_PACK>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        math::dest_bank          = 1 - math::dest_bank;
    }
}

/**
 * @brief Initialize the math/pack dest semaphore and reset the dest section.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    using namespace hal_llk;
    // Wait for the packer to drain every section of a previous run before claiming dest.
    while (hal::sync::semaphore::read<SEM_MATH_PACK>() > 0)
    {
    }
    hal::sync::wait::stall<hal::sync::StallTarget::Sync, hal::sync::StallCondition::MathIdle | hal::sync::StallCondition::SfpuIdle>();
    if constexpr (Dst == DstSync::SyncFull)
    {
        hal::sync::semaphore::init<0 /* Initial */, 1 /* Maximum */, SEM_MATH_PACK>();
    }
    else
    {
        hal::sync::semaphore::init<0 /* Initial */, 2 /* Maximum */, SEM_MATH_PACK>();
    }
    math::dest_bank = 0;
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0, 0>();
}
