// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL-based implementation of the common math-thread configuration and Dest synchronization (Blackhole).

#pragma once

#include <cstdint>

#include "hal_llk_common.h"

namespace hal_llk
{

// Row of the current Dest section (0 or the second half).
inline std::uint32_t math_dest_section_base()
{
    return (dest_offset_id != 0) ? DEST_HALF_BASE_ROW : 0;
}

// One math address-modifier slot (ADDR_MOD k) for SrcA, SrcB, Dest and fidelity counters.
struct MathAddrMod
{
    std::uint32_t srca_incr  = 0;
    bool srca_clear          = false;
    std::uint32_t srcb_incr  = 0;
    bool srcb_clear          = false;
    std::uint32_t dest_incr  = 0;
    bool dest_clear          = false;
    bool dest_cr             = false; // live counter reloaded from carry/reset + increment
    bool dest_c_to_cr        = false; // copy the (post-increment) live counter to carry/reset
    std::uint32_t fidel_incr = 0;
    bool fidel_clear         = false;
};

template <hal::cfg::Sec S>
inline void program_math_addrmod(const MathAddrMod& m)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::AddrMod[cfg::SrcA].Incr, S>(m.srca_incr),
        set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        set<cfg::AddrMod[cfg::SrcA].Clear, S>(m.srca_clear ? 1 : 0),
        set<cfg::AddrMod[cfg::SrcB].Incr, S>(m.srcb_incr),
        set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        set<cfg::AddrMod[cfg::SrcB].Clear, S>(m.srcb_clear ? 1 : 0));
    cfg::write<cfg::Access::TensixCfgUnit>(set<cfg::AddrMod[cfg::SrcA].Incr2, S, 0>(), set<cfg::AddrMod[cfg::SrcB].Incr2, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::AddrMod[cfg::Dest].Incr, S>(m.dest_incr),
        set<cfg::AddrMod[cfg::Dest].CR, S>(m.dest_cr ? 1 : 0),
        set<cfg::AddrMod[cfg::Dest].Clear, S>(m.dest_clear ? 1 : 0),
        set<cfg::AddrMod[cfg::Dest].CToCR, S>(m.dest_c_to_cr ? 1 : 0),
        set<cfg::AddrMod[cfg::Fidelity].Incr, S>(m.fidel_incr),
        set<cfg::AddrMod[cfg::Fidelity].Clear, S>(m.fidel_clear ? 1 : 0));
    cfg::write<cfg::Access::TensixCfgUnit>(set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

} // namespace hal_llk

/**
 * @brief Configure the math (FPU) thread's ALU control: FP32 Dest accumulation and INT8 math.
 *
 * The SrcA/SrcB ALU formats are programmed by the unpacker on Blackhole (see _llk_unpack_hw_configure_).
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg = hal::cfg;
    using cfg::set;
    using namespace hal::sync;

    wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();

    const std::uint32_t int8_math = (hal_llk::is_int8_math_format(srca_data_format) || hal_llk::is_int8_math_format(srcb_data_format)) ? 1 : 0;
    cfg::write<cfg::Access::TensixCfgUnit>(
        set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math));
}

/**
 * @brief Block the math thread until the current Dest section has been released by the packer.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    using namespace hal::sync;
    wait::semaphore<StallTarget::Math, SemaphoreCondition::WhileMaximum, hal_llk::MATH_PACK_SEMAPHORE>();
}

/**
 * @brief Hand the finished Dest section to the packer and, in SyncHalf mode, switch to the other half.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    // Every Dest write of this section must be complete before the packer is signalled.
    wait::stall<StallTarget::Sync, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    semaphore::post<hal_llk::MATH_PACK_SEMAPHORE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_offset_id           = 1 - dest_offset_id;
        wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(hal_llk::math_dest_section_base());
    }
}

/**
 * @brief Initialize the math/pack Dest semaphore and select the first Dest section.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace cfg = hal::cfg;
    using namespace hal::sync;

    // Wait until any previous kernel has released Dest.
    while (semaphore::read<hal_llk::MATH_PACK_SEMAPHORE>() > 0)
    {
    }
    if constexpr (Dst == DstSync::SyncFull)
    {
        semaphore::init<0, 1, hal_llk::MATH_PACK_SEMAPHORE>();
    }
    else
    {
        semaphore::init<0, 2, hal_llk::MATH_PACK_SEMAPHORE>();
    }

    dest_offset_id           = 0;
    math_sync_tile_dst_index = 0;
    wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, 0>();
}
