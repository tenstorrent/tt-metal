// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_hal_common.h"

namespace hal_llk
{

// One math ADDR_MOD slot: how the SrcA/SrcB/Dest/fidelity counters move after an instruction.
struct MathAddrMod
{
    std::uint32_t srca_incr  = 0;
    std::uint32_t srca_clear = 0;
    std::uint32_t srcb_incr  = 0;
    std::uint32_t srcb_clear = 0;
    std::uint32_t dest_incr  = 0;
    std::uint32_t dest_clear = 0;
    std::uint32_t dest_cr    = 0; // increment relative to the carry/reset value
    std::uint32_t dest_c2cr  = 0; // save the incremented dest counter into carry/reset
    std::uint32_t fid_incr   = 0;
    std::uint32_t fid_clear  = 0;
};

// Program math ADDR_MOD slot S. Every field of the slot's thread-CFG words is supplied, so the
// grouped SETC16 writes leave no stale bits behind.
template <hal::cfg::Sec S, MathAddrMod M>
inline void set_math_addr_mod()
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clear>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clear>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr2, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clear>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c2cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clear>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// Point the math thread at a dest row (section base + tile offset).
inline void set_math_dest_offset(const std::uint32_t row)
{
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0>(row);
}

} // namespace hal_llk

/**
 * @brief Configure the math ALU control: INT8 math, FP32 dest accumulation, source zero flags.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();

    const std::uint32_t int8_math = (hal_llk::is_int8_or_int32(srca_data_format) || hal_llk::is_int8_or_int32(srcb_data_format)) ? 1 : 0;
    // Zero-substitution must stay disabled only for 16-bit integer (UInt16) sources.
    const std::uint32_t keep_src_zeros =
        (srca_data_format == static_cast<std::uint32_t>(DataFormat::UInt16) || srcb_data_format == static_cast<std::uint32_t>(DataFormat::UInt16)) ? 1 : 0;

    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math));
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(keep_src_zeros);
}

/**
 * @brief Block math until the packer has released a dest section.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    namespace sync = hal::sync;
    sync::wait::semaphore<sync::StallTarget::Math | sync::StallTarget::Sfpu, sync::SemaphoreCondition::WhileMaximum, hal_llk::MATH_PACK_SEM>();
}

/**
 * @brief Hand the finished dest section to the packer and, in half-sync mode, move to the other half.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    namespace sync = hal::sync;

    // Post only after every math/SFPU write to the section has landed.
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::semaphore::post<hal_llk::MATH_PACK_SEM>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        ckernel::dest_offset_id  = 1 - ckernel::dest_offset_id;
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
        hal_llk::set_math_dest_offset(hal_llk::dest_section_base());
    }
}

/**
 * @brief Initialise the math <-> pack dest handshake and select dest section 0.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace sync = hal::sync;

    // Let a previous kernel's packer drain everything before claiming dest.
    while (sync::semaphore::read<hal_llk::MATH_PACK_SEM>() > 0)
    {
    }
    if constexpr (Dst == DstSync::SyncFull)
    {
        sync::semaphore::init<0 /* Initial */, 1 /* Maximum */, hal_llk::MATH_PACK_SEM>();
    }
    else
    {
        sync::semaphore::init<0 /* Initial */, 2 /* Maximum */, hal_llk::MATH_PACK_SEM>();
    }

    ckernel::dest_offset_id = 0;
    hal_llk::set_math_dest_offset(0);
}
