// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_welfords_sfpu.h, ported except SFPU. Same Tensix instruction sequence
// as production.

#pragma once
#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu_types.h"
#include "sfpu/ckernel_sfpu_welfords.h"

// local function declarations
inline void welfords_sfpu_configure_addrmod()
{
    hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S7 /*ADDR_MOD_7*/, 0>();
}

inline void welfords_sfpu_configure_mop();

template <DstSync Dst>
inline void _llk_math_welfords_sfpu_start_(const std::uint32_t dst_index)
{
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    hal::sync::wait::stall<hal::sync::StallTarget::Sfpu, hal::sync::StallCondition::MathIdle>();
}

inline void _llk_math_welfords_sfpu_done_()
{
    hal_port::clear_dst_reg_addr();

    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::SfpuIdle>();
}

inline void _llk_math_welfords_sfpu_inc_dst_face_addr_()
{
    hal_port::inc_dst_addr_8();
    hal_port::inc_dst_addr_8();
}

inline void _llk_math_welfords_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    welfords_sfpu_configure_addrmod();
    hal_port::reset_counters_abd_f();
    _program_welfords_replay_buffer_();
}

/**
 * @brief Configure SFPU constants, address modifiers, and counters for two-pass statistics.
 *
 * @note Call @ref _two_pass_clear_stats_ before accumulating a new population; initialisation does not clear the
 * running register state.
 */
inline void _llk_math_two_pass_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    welfords_sfpu_configure_addrmod();
    hal_port::reset_counters_abd_f();
}

/**
 * Re-establish MATH address mods and MOP state for the SFPU Welford path after an
 * arbitrary MATH/FPU op (e.g. eltwise binary scalar multiply). Does not touch the
 * SFPU replay buffer or clear running mean/M2 in LREG4/5.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_math_welfords_sfpu_reinit_(const std::uint32_t num_faces, const std::uint32_t dst_format)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    eltwise_unary_configure_addrmod<DataCopyType::A2D>(dst_format);
    eltwise_unary_configure_mop<DataCopyType::A2D, is_fp32_dest_acc_en>(
        p_mova2d::MOV_8_ROWS, 16, ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), dst_format);
    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): the word holds both SrcA and SrcB disables.
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
        hal::cfg::set<hal::cfg::ClrDvalid::SrcA_Disable, hal::cfg::Sec::S0, 0>(), hal::cfg::set<hal::cfg::ClrDvalid::SrcB_Disable, hal::cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f();
}

inline void _llk_math_welfords_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}
