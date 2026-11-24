// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_unary_datacopy_softmax_k.h. Same Tensix
// instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/move.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

inline void _llk_math_eltwise_unary_datacopy_softmax_k_(const std::uint32_t dst_index)
{
    namespace move = hal::move;
    namespace rwc  = hal::counters::rwc;

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Copy row 0 from SrcB, then broadcast its first scalar into DEST row 8.
    // TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_1_ROW, 0)
    move::run<move::Transfer<move::SrcB, move::Dst> {
        .source_row = p_movb2d::SRC_ZERO_OFFSET, .destination_row = 0, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::One}>();
    // TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_1_ROW_D0_BRCST, 0)
    move::run<move::Transfer<move::SrcB, move::Dst> {
        .source_row      = p_movb2d::SRC_ZERO_OFFSET,
        .destination_row = 0,
        .address_mode    = ADDR_MOD_0,
        .number_of_rows  = move::rows::One,
        .broadcast       = move::Broadcast::Column0}>();
    // TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0)
    rwc::set<rwc::MathCounter {.flip = rwc::Src::A | rwc::Src::B}>();

    hal_port::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_unary_datacopy_softmax_k_init_()
{
    namespace cfg = hal::cfg;

    hal_port::set_math_addr_mod<cfg::Sec::S0, hal_port::MathAddrMod {.dest_incr = 8}>();

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f();
}
