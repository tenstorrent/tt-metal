// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

inline void llk_math_top32_rm_configure_addrmod() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);

    addr_mod_t{
        .srca = {.incr = 1},
        .srcb = {.incr = 0},
        .dest = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
}

template <bool is_fp32_dest_acc_en, bool is_int_fpu_en = false>
inline void llk_math_top32_rm_configure_mop(const std::uint32_t total_rows, const std::uint32_t dst_format) {
    const std::uint32_t innerloop = 2;
    const std::uint32_t outerloop = 1;

    if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) ||
        (dst_format == to_underlying(DataFormat::UInt8))) {
        ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
        // tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    } else {
        ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
        // tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
}

template <bool is_fp32_dest_acc_en, bool is_int_fpu_en = false>
inline void _llk_math_top32_rm_init_(const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255) {
    llk_math_top32_rm_configure_addrmod();

    const std::uint32_t num_rows = 8;
    llk_math_top32_rm_configure_mop<is_fp32_dest_acc_en, is_int_fpu_en>(num_rows, dst_format);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool unpack_to_dest = false>
inline void _llk_math_top32_rm_(
    const std::uint32_t dst_index,
    const std::uint32_t src_format,
    const std::uint32_t dst_format,
    const std::uint32_t num_faces) {
    if (unpack_to_dest && is_32bit_input(src_format, dst_format)) {
        math_unpack_to_dest_math_ready();
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        math::math_unpack_to_dest_tile_ready();

        // Due to bug in Blackhole Tensix (more details in budabackend/#2730) when an event with side effect of clearing
        // DEST zero flags (such as Unpack-to-dest or RISC-to-dest) and a ZEROACC instruction from packer occur in the
        // same cycle, zero flags clearing is dropped. To mitigate that, we issue additional zero flag clear instruction
        // immediately after unpack tile to dest is done. RISC-to-dest event is not currently used.

        const std::uint32_t dst_format_masked = masked_data_format(dst_format);
        const int clear_fp32 = static_cast<int>(
            dst_format_masked == (std::uint32_t)DataFormat::Float32 ||
            dst_format_masked == (std::uint32_t)DataFormat::Int32 ||
            dst_format_masked == (std::uint32_t)DataFormat::UInt32);
        const std::uint32_t tiles_per_bank = clear_fp32 ? 4 : 8;
        const std::uint32_t local_tile = dst_index & (tiles_per_bank - 1);
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < num_faces; i++) {
            TT_ZEROACC(
                p_zeroacc::CLR_16,
                clear_fp32,
                1 /*clear zero flags*/,
                ADDR_MOD_3,
                get_dest_index_in_faces(local_tile, i));
        }
    } else {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        // always do all 4 faces because the lower faces are set to -inf for sorting
        for (std::uint32_t i = 0; i < 4; i++) {
            ckernel_template::run();
        }
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);

        math::clear_dst_reg_addr();
    }
}
