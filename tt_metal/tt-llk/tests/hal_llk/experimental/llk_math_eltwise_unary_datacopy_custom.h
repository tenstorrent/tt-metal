// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_unary_datacopy_custom.h. Same instruction
// sequence and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/move.h"
#include "hal/tensix_counters.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

// Uses ADDR_MOD_4 only ({srca=8, dest=8}), preserving matmul's ADDR_MOD 0,1,2
// and reduce's ADDR_MOD 2,3. ADDR_MOD_4 is a "don't care" — matmul restores it
// on next reinit. Assumes full 32x32 tile (4 faces, 8 rows each).
inline void _llk_math_eltwise_unary_datacopy_custom_(const std::uint32_t dest_index = 0)
{
    namespace cfg  = hal::cfg;
    namespace rwc  = hal::counters::rwc;
    namespace move = hal::move;

    hal_port::set_math_addr_mod<cfg::Sec::S4, hal_port::MathAddrMod {.srca_incr = 8, .dest_incr = 8}>(); // ADDR_MOD_4

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    hal_port::reset_counters_abd_f(); // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD_F)
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dest_index);

    // TTI_MOVA2D(0, 0, ADDR_MOD_4, MOV_8_ROWS, 0)
    constexpr move::Transfer<move::SrcA, move::Dst> mova2d_8_rows {
        .source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_4, .number_of_rows = move::rows::Eight};
#pragma GCC unroll 8
    for (int i = 0; i < 8; i++)
    {
        move::run<mova2d_8_rows>();
    }
    // TTI_SETRWC(CLR_A, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A}>();
}
