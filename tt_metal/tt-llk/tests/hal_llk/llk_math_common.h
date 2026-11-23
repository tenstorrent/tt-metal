// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_common.h (every function) and the common/inc/cmath_common.h
// helpers it and llk_math_eltwise_binary.h reach. Same Tensix instruction sequence as production.

#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h" // ckernel::math globals (src_zero_flag_*) and pure helpers only
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/move.h"
#include "hal/sync.h"

using namespace ckernel::math;

namespace hal_port
{

namespace cfg  = hal::cfg;
namespace sync = hal::sync;
namespace rwc  = hal::counters::rwc;

// ckernel::semaphore::MATH_PACK
inline constexpr sync::Semaphore MATH_PACK_SEM = sync::Semaphore::S1;

// ---------------------------------------------------------------------------------------------
// Src zero-substitution flag (cmath_common.h). The tracking variables are the production ones.
// ---------------------------------------------------------------------------------------------

// The one writer, out-of-line as in production. No STALLWAIT, as in production.
inline __attribute__((noinline)) void _apply_src_zero_flag_(const std::uint32_t value)
{
    ckernel::math::src_zero_flag_hw = value;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(value);
}

inline void _configure_default_zero_flag_state_()
{
    const std::uint32_t value = requires_disabled_src_zero_flag(ckernel::math::src_zero_flag_srca_fmt, ckernel::math::src_zero_flag_srcb_fmt) ? 1u : 0u;
    if (ckernel::math::src_zero_flag_hw == value)
    {
        return;
    }
    _apply_src_zero_flag_(value);
}

// ---------------------------------------------------------------------------------------------
// cmath_common.h helpers
// ---------------------------------------------------------------------------------------------

// TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, setrwc) with a compile-time mask: SET_ABD_F = A, B, D and fidelity.
inline void reset_counters_abd_f()
{
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .clear_fidelity = true}>();
}

// MOVD2A: Dest face rows -> SrcA rows. Production passes addrmod at runtime but every caller passes a constant.
template <std::uint8_t addrmod>
inline void move_d2a_fixed_face()
{
    namespace move = hal::move;
    // Wait until the Matrix Unit owns the bank MOVD2A writes. No Matrix Unit drain: when the instruction that released
    // SrcA (ELW or SETRWC with CLR_A) is still in the unit, the condition waits for the bank it moves to (MOVD2A.md).
    // Dst reads by MOVD2A are interlocked against earlier Matrix Unit writes (Dst.md).
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SrcAValid>();
    // TTI_MOVD2A(dest_32b_lo, src = SrcA row, addrmod, instr_mod, dst = Dest row)
    move::run<move::Transfer<move::Dst, move::SrcA> {
        .source_row = 0, .destination_row = p_mova2d::MATH_HALO_ROWS + 0, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcA> {
        .source_row = 4, .destination_row = p_mova2d::MATH_HALO_ROWS + 4, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcA> {
        .source_row = 8, .destination_row = p_mova2d::MATH_HALO_ROWS + 8, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcA> {
        .source_row = 12, .destination_row = p_mova2d::MATH_HALO_ROWS + 12, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
}

// MOVD2B: Dest face rows -> SrcB rows.
template <std::uint8_t addrmod>
inline void move_d2b_fixed_face()
{
    namespace move = hal::move;
    // Keeps the Matrix Unit drain: MOVD2B.md, unlike MOVD2A.md, does not say the SrcB-valid wait follows an in-flight release.
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 0, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 0, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 4, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 4, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 8, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 8, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 12, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 12, .address_mode = addrmod, .number_of_rows = move::rows::Four}>();
}

inline void set_math_semaphores()
{
    // Tell packer that it has something to pack: t6_semaphore_post<MATH | WAIT_SFPU>(MATH_PACK)
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    sync::semaphore::post<MATH_PACK_SEM>();
}

template <DstTileShape tile_shape, UnpackDestination unpack_destination>
inline void set_dst_write_addr(std::uint32_t tile_index)
{
    static_assert(
        tile_shape == DstTileShape::Tile32x32 || tile_shape == DstTileShape::Tile32x16 || tile_shape == DstTileShape::Tile16x16, "Invalid tile shape");
    static_assert(DstTileShape::Tile32x32 == 0, "DstTileShape::Tile32x32 must equal 0");
    static_assert(DstTileShape::Tile32x16 == 1, "DstTileShape::Tile32x16 must equal 1");
    static_assert(DstTileShape::Tile16x16 == 2, "DstTileShape::Tile16x16 must equal 2");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x32] == 6, "DstTileSizeLog2[Tile32x32] must equal 6");
    static_assert(DstTileSizeLog2[DstTileShape::Tile32x16] == 5, "DstTileSizeLog2[Tile32x16] must equal 5");
    static_assert(DstTileSizeLog2[DstTileShape::Tile16x16] == 4, "DstTileSizeLog2[Tile16x16] must equal 4");

    std::uint32_t dst_index = tile_index << DstTileSizeLog2[tile_shape];
    dst_index               = dst_index + ckernel::get_dest_buffer_base();
    if constexpr (unpack_destination == UnpackDestination::DestReg)
    {
        ckernel::mailbox_write(ThreadId::UnpackThreadId, dst_index); // HAL-GAP: no HAL API for the thread mailbox
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dst_index);
    }
}

// TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_D)
inline void clear_dst_reg_addr()
{
    rwc::set<rwc::MathCounter {.dest = rwc::Set(0)}>();
}

inline void math_dest_wait()
{
    sync::wait::semaphore<sync::StallTarget::Math | sync::StallTarget::Sfpu | sync::StallTarget::Sync, sync::SemaphoreCondition::WhileMaximum, MATH_PACK_SEM>();
}

inline void dest_section_flip()
{
    ckernel::update_dest_offset_id();
    std::uint32_t base_addr = ckernel::get_dest_buffer_base();
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(base_addr);
}

template <DstStart Dst>
inline void set_dest_section_base()
{
    if constexpr (Dst == DstStart::StartZero)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, 0>();
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0, DEST_REGISTER_HALF_SIZE>();
    }
}

// Shared body of the three _llk_math_reconfig_data_format*_ functions (production repeats it inline).
inline __attribute__((always_inline)) void reconfig_int8_math_enabled(const bool int8_math_enabled)
{
    // Only INT8_math_enabled is written here; it is FPU-only, so draining the FPU (MATH) suffices -- no WAIT_SFPU.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math_enabled);
}

} // namespace hal_port

/**
 * @brief Configure the math (FPU) thread's ALU control registers for the given source data formats.
 */
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // SFPU_Fp32_enabled (written below) is read by SFPLOAD/SFPSTORE (MOD0_FMT_SRCB), so the SFPU must drain too.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle>();
    // Configure ZEROACC to auto-detect destination bank (non-legacy mode).
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
    std::uint32_t int8_math_enabled = is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format);
    // INT8 math, FP32 dest accumulation and SFPU FP32 share ALU_ACC_CTRL word 0 (production: three RMWs).
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(int8_math_enabled),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>(),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, is_fp32_dest_acc_en ? 1 : 0>());
    LLK_ASSERT_BLOCK(llk_assert_fp32_dest_acc_en = is_fp32_dest_acc_en);

    // Establish the no-override baseline for the SrcA/SrcB ALU format-select fields (ADDR32=0): clear
    // SrcA_val/override and SrcB_val/override. Dstacc fields in the same word are owned by the pack thread.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());

    // Establish the operand-driven baseline for the Src zero-substitution flag; refresh both format caches first.
    ckernel::math::src_zero_flag_srca_fmt = srca_data_format;
    ckernel::math::src_zero_flag_srcb_fmt = srcb_data_format;
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Enable or disable the destination read-address remap (stride-of-16) used by untilize mode.
 */
inline void _llk_math_reconfig_remap_(const bool remap_enable)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // Need to wait for all DEST accesses to be finished before changing
    // remap_addrs and swizzle_32b bits
    ckernel::tensix_sync(); // HAL-GAP: no public HAL API for the PC-buffer tensix_sync() RISC fence
    while (sync::semaphore::read<hal_port::MATH_PACK_SEM>() > 0)
    {
    } // Wait for previous packs to finish before claiming all dest

    // Untilize mode needs dest read access with a stride of 16
    // remap_addrs and swizzle_32b share DEST_ACCESS_CFG byte 0 (production: two RMWs).
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::DestAccessCfg::remap_addrs, cfg::Sec::S0>(remap_enable), cfg::set<cfg::DestAccessCfg::swizzle_32b, cfg::Sec::S0>(remap_enable));
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 */
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    // These lightweight functions for sync with packer imply
    // no mode change - entire epoch is either double buffer or single buffer
    hal_port::math_dest_wait();
}

/**
 * @brief Signal completion of a destination section by setting the math semaphores, flipping banks in half-sync mode.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    hal_port::set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        hal_port::dest_section_flip();
    }
}

/**
 * @brief Clear the acquired DST section from MATH instead of clearing it on PACK release.
 *
 * Call immediately after acquiring the section, before any DST writes, and use PACK release without clearing throughout the kernel.
 * DST writes must stay between acquire and commit; commit drains them before the next acquisition.
 * Unpack-to-DST must wait for MATH's ready signal; there must be no independent DST writer.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_clear_dest_section_()
{
    namespace dst  = hal::dst;
    namespace sync = hal::sync;

    constexpr dst::DestWidth width = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;
    // Startup or the preceding commit has already drained this thread's DST writes.
    if constexpr (Dst == DstSync::SyncFull)
    {
        // TTI_ZEROACC(CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0)
        dst::run<dst::Zero {.scope = dst::ZeroScope::All, .index = 0, .width = width, .address_mode = ADDR_MOD_1}>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        // TT_ZEROACC(CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2)
        dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = dest_offset_id % 2, .width = width, .address_mode = ADDR_MOD_1});
    }
    // Do not let SFPU writes or an unpack-to-DST ready signal overtake the clear.
    sync::wait::stall<sync::StallTarget::Sfpu | sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
}

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    namespace sync = hal::sync;

    ckernel::tensix_sync(); // HAL-GAP: no public HAL API for the PC-buffer tensix_sync() RISC fence
    while (sync::semaphore::read<hal_port::MATH_PACK_SEM>() > 0)
    {
    } // Wait for previous packs to finish before claiming all dest
    if constexpr (Dst == DstSync::SyncFull)
    {
        sync::semaphore::init<0 /*Initial*/, 1 /*Maximum*/, hal_port::MATH_PACK_SEM>();
        ckernel::reset_dest_offset_id();
        hal_port::set_dest_section_base<StartZero>();
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        sync::semaphore::init<0 /*Initial*/, 2 /*Maximum*/, hal_port::MATH_PACK_SEM>();
        ckernel::reset_dest_offset_id();
        hal_port::set_dest_section_base<StartZero>();
    }
}

// Following functions do not need to program ALU_FORMAT_SPEC_REG0_SrcA/ALU_FORMAT_SPEC_REG1_SrcB
// for blackhole since ALU format is inferred

/**
 * @brief Reconfigure the math thread for a new source A data format (re-derives INT8_math_enabled unless skip_int8).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_srca_(const std::uint32_t srca_data_format)
{
    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !(is_int8_or_int32_format(srca_data_format)),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        hal_port::reconfig_int8_math_enabled(is_int8_or_int32_format(srca_data_format));
    }

    // Re-establish the operand-driven baseline for the new format(s); refresh the cache(s) unconditionally.
    ckernel::math::src_zero_flag_srca_fmt = srca_data_format;
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Reconfigure the math thread for a new source B data format (re-derives INT8_math_enabled unless skip_int8).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_srcb_(const std::uint32_t srcb_data_format)
{
    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !(is_int8_or_int32_format(srcb_data_format)),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        hal_port::reconfig_int8_math_enabled(is_int8_or_int32_format(srcb_data_format));
    }

    // Re-establish the operand-driven baseline for the new format(s); refresh the cache(s) unconditionally.
    ckernel::math::src_zero_flag_srcb_fmt = srcb_data_format;
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Reconfigure the math thread for new source A and source B data formats (re-derives INT8_math_enabled unless skip_int8).
 */
template <bool is_fp32_dest_acc_en, bool skip_int8 = false>
inline void _llk_math_reconfig_data_format_(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
{
    if constexpr (!skip_int8)
    {
        LLK_ASSERT(
            is_fp32_dest_acc_en || !(is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format)),
            "Reconfiguring math to/from Int8/UInt8/Int32 formats requires FP32 Dest mode enabled");
        hal_port::reconfig_int8_math_enabled(is_int8_or_int32_format(srca_data_format) || is_int8_or_int32_format(srcb_data_format));
    }

    // Re-establish the operand-driven baseline for the new format(s); refresh the cache(s) unconditionally.
    ckernel::math::src_zero_flag_srca_fmt = srca_data_format;
    ckernel::math::src_zero_flag_srcb_fmt = srcb_data_format;
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Read the FPU sticky special-value flags (e.g. NaN/Inf detection) accumulated since the last clear.
 */
inline std::uint32_t _llk_math_get_compute_special_value_flags_()
{
    return reg_read(RISCV_DEBUG_REG_FPU_STICKY_BITS); // HAL-GAP: no public HAL API for RISC debug registers
}

/**
 * @brief Clear the FPU sticky special-value flags register.
 */
inline void _llk_math_clear_compute_special_value_flags_()
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0); // HAL-GAP: no public HAL API for RISC debug registers
}
