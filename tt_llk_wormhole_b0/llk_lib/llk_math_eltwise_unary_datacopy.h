// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

// local function declarations
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format);

template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_(const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format)
{
    if (unpack_to_dest && is_32bit_input(src_format, dst_format))
    {
        math_unpack_to_dest_math_ready();
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst_index);
        math::math_unpack_to_dest_tile_ready();

        if constexpr (src_b_bcast_type == BroadcastType::ROW)
        {
            // workarounds for hi/lo D2B/B2D on BH (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            TTI_SETDVALID(0b10);

            // move back to B and broadcast in 2 parts, first hi16 bits then lo16 bits
            constexpr int dest_32b_hi = 0;
            constexpr int dest_32b_lo = 1;

            // move hi bits D2B
            TTI_MOVD2B(dest_32b_hi, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
            TTI_MOVD2B(dest_32b_hi, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 16);

            // broadcast hi bits B2D
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 0);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 8);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 32);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 40);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 16);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 24);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 48);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 56);

            // move lo bits D2B
            TTI_MOVD2B(dest_32b_lo, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
            TTI_MOVD2B(dest_32b_lo, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 16);

            // broadcast lo bits B2D
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 0);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 8);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 32);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 40);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 16);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 24);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 48);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST, 56);

            // restore fp32 mode
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
        {
            // workarounds for hi/lo D2B/B2D on BH (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            TTI_SETDVALID(0b10);

            // move back to B and broadcast in 2 parts, first hi16 bits then lo16 bits
            constexpr int dest_32b_hi = 0;
            constexpr int dest_32b_lo = 1;

            // move hi bits D2B
            TTI_MOVD2B(dest_32b_hi, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);

            // broadcast hi bits B2D
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 0);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 8);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 16);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 24);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 32);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 40);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 48);
            TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 56);

            // move lo bits D2B
            TTI_MOVD2B(dest_32b_lo, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);

            // broadcast lo bits B2D
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 0);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 8);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 16);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 24);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 32);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 40);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 48);
            TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_3, p_movb2d::MOV_8_ROW_BRCST_D0_BRCST, 56);

            // restore fp32 mode
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            TTI_CLEARDVALID(0b10, 0);
        }
        else if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // workarounds for hi/lo D2B/B2D on BH (Issue #449)
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1); // Do not 0 out ints
            TTI_SETDVALID(0b10);

#pragma GCC unroll 2
            for (int offset = 0; offset < 2; ++offset)
            {
#pragma GCC unroll 2
                for (int dst_32b_hi_lo_idx = 0; dst_32b_hi_lo_idx < 2; ++dst_32b_hi_lo_idx)
                {
                    // move hi bits D2B
                    TTI_MOVD2B(dst_32b_hi_lo_idx, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 0);
                    TTI_MOVD2B(dst_32b_hi_lo_idx, p_movd2b::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 4);
                    TTI_MOVD2B(dst_32b_hi_lo_idx, p_movd2b::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 8);
                    TTI_MOVD2B(dst_32b_hi_lo_idx, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, offset * 32 + 12);

                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 0);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 4);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 8);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 12);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 16);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 20);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 24);
                    TTI_MOVB2D(dst_32b_hi_lo_idx, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS_D0_BRCST, offset * 32 + 28);
                }
            }

            // restore fp32 mode

            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
            TTI_CLEARDVALID(0b10, 0);
        }
    }
    else
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

        if constexpr (type == A2D)
        {
            ckernel_template::run();
        }
        else if constexpr (type == B2D)
        {
            if constexpr (src_b_bcast_type == BroadcastType::COL)
            {
                // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0);
            }
            else
            {
                ckernel_template::run();
            }
        }

        math::clear_dst_reg_addr();
    }
}

template <DataCopyType type, BroadcastType bcast_type = BroadcastType::NONE>
inline void eltwise_unary_configure_addrmod(const std::uint32_t dst_format)
{
    // Use srcA for data movement
    if constexpr (type == A2D)
    {
        addr_mod_t {
            .srca = {.incr = 1},
            .srcb = {.incr = 0},
            .dest = {.incr = 1},
        }
            .set(ADDR_MOD_0);

        // Just unpack into A and move to Dest
        addr_mod_t {
            .srca = {.incr = 8},
            .srcb = {.incr = 0},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_2);

        if constexpr (bcast_type != BroadcastType::NONE)
        {
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 0},
            }
                .set(ADDR_MOD_3);
        }
    }
    else
    {
        if constexpr (bcast_type == BroadcastType::ROW || bcast_type == BroadcastType::SCALAR)
        {
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 1},
            }
                .set(ADDR_MOD_0);

            // Just unpack into B and move to Dest
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_2);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 0},
                .srcb = {.incr = 1},
                .dest = {.incr = 1},
            }
                .set(ADDR_MOD_0);

            // Just unpack into B and move to Dest
            if (dst_format == to_underlying(DataFormat::UInt16)) // UInt16 case needs to use MOVB2D, which is 4 rows per op
            {
                addr_mod_t {
                    .srca = {.incr = 0},
                    .srcb = {.incr = 4},
                    .dest = {.incr = 4},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 0},
                    .srcb = {.incr = 8},
                    .dest = {.incr = 8},
                }
                    .set(ADDR_MOD_2);
            }
        }
    }
}

template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType bcast_type = BroadcastType::NONE, bool is_int_fpu_en = false>
inline void eltwise_unary_configure_mop(std::uint32_t rows_per_inst, std::uint32_t total_rows, const std::uint32_t num_faces, const std::uint32_t dst_format)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // always move 32x32 tile, packed as 16x16x4

    if constexpr (type == A2D)
    {
        std::uint32_t innerloop = (rows_per_inst == p_mova2d::MOV_1_ROW) ? total_rows : (total_rows >> 3);
        std::uint32_t outerloop = num_faces;

        if (((is_fp32_dest_acc_en || is_int_fpu_en) && !(dst_format == to_underlying(DataFormat::UInt16))) || (dst_format == to_underlying(DataFormat::UInt8)))
        {
            // use elwadd to handle unpacking data into src A as fp16, but dest is in fp32 mode OR to handle uint8 datums
            ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
    else if constexpr (type == B2D)
    {
        std::uint32_t addr_mod  = (rows_per_inst == p_movb2d::MOV_1_ROW) ? ADDR_MOD_0 : ADDR_MOD_2;
        std::uint32_t innerloop = (rows_per_inst == p_movb2d::MOV_1_ROW) ? total_rows : (total_rows >> 2);
        std::uint32_t outerloop = 4;
        auto broadcast_type     = p_movb2d::MOV_1_ROW; // No broadcast;

        if constexpr (bcast_type == BroadcastType::COL)
        {
            innerloop = 16 >> 3; // elwadd produces 8 rows per op
            // The mop only runs for 2 outer loops and mop is called twice for col broadcast
            outerloop = 2;
            // ELWADD with zeros will be used for non UInt16 case, since it moves 8 rows per cycle
            broadcast_type = p_elwise::SRCB_BCAST_COL;
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                innerloop      = 16 >> 2; // movb2d produces 4 rows per op
                broadcast_type = p_movb2d::MOV_4_ROWS_D0_BRCST;
            }
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            innerloop      = (total_rows >> 3);
            broadcast_type = p_movb2d::MOV_8_ROW_BRCST;
        }
        else if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            outerloop      = 1;
            innerloop      = num_faces * (total_rows >> 3);
            broadcast_type = p_elwise::SRCB_BCAST_ALL;
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                broadcast_type = p_movb2d::MOV_8_ROW_BRCST_D0_BRCST;
            }
        }

        if constexpr (bcast_type == BroadcastType::SCALAR)
        {
            if (dst_format == to_underlying(DataFormat::UInt16))
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0));
                tmp.program();
            }
            else
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0));
                tmp.program();
            }
        }
        else if constexpr (bcast_type == BroadcastType::COL)
        {
            if (dst_format ==
                to_underlying(DataFormat::UInt16)) // UInt16 case needs to use MOVB2D because for ELWADD FPU interprets some numbers as a float with exp 0
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
                tmp.set_end_op(TT_OP_SETRWC(0, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
                tmp.program();
            }
            else // ELWADD is used for non UInt16 case, since it moves 8 rows per cycle
            {
                ckernel_template tmp(outerloop, innerloop, TT_OP_ELWADD(0, 0, broadcast_type, addr_mod, 0));
                tmp.set_end_op(TT_OP_SETRWC(0, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
                tmp.program();
            }
        }
        else if constexpr (bcast_type == BroadcastType::ROW)
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, broadcast_type, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, TT_OP_MOVB2D(0, 0, addr_mod, rows_per_inst, 0));
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_B));
            tmp.program();
        }
    }
}

template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool is_int_fpu_en = false>
inline void _llk_math_eltwise_unary_datacopy_init_(const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    eltwise_unary_configure_addrmod<type, src_b_bcast_type>(dst_format);

    if constexpr (type == A2D && src_b_bcast_type == BroadcastType::NONE)
    {
        eltwise_unary_configure_mop<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(p_mova2d::MOV_8_ROWS, 16, num_faces, dst_format);
    }
    else if constexpr (type == B2D)
    {
        eltwise_unary_configure_mop<type, false, src_b_bcast_type>(p_movb2d::MOV_4_ROWS, 16, num_faces, dst_format);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_uninit_()
{
    // clear debug feature disable
    if constexpr (src_b_bcast_type != BroadcastType::NONE && unpack_to_dest)
    {
        _llk_math_dbg_feature_enable_();
    }
}

/*************************************************************************
 * LLK MATH FAST TILIZE (Tilize single input using both unpackers and packer)
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim and num_units must match the ones given to the unpacker (all unit_dim usage notes from the unpacker also apply here)
 * dst_index is the index of the tile inside the destination register to write to
 * both dest modes are supported (although 32 bit mode is supported by intentionally ignoring it for both math and pack unless src regs are TF32)
 * only DstSync::SyncHalf is supported
 * tiles are split across halves of the active dest bank (effectively quarters since DstSync::SyncHalf)
 * so nothing except fast tilize should be using that dest bank
 *************************************************************************/

inline void _llk_math_fast_tilize_addrmod_config_(const std::uint32_t unpack_dst_format, const std::uint32_t unit_dim)
{
    // standard addrmod that follows MOVB2D
    addr_mod_t {
        .srcb = {.incr = 4},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_1);

    // standard addrmod that follows MOVA2D
    addr_mod_t {
        .srca = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

    // next two addrmods are mostly used for jumping to and from the offset for the bottom faces
    // offset for the bottom faces is always half the number of rows in the dest bank (512 / 2 for 16bit and 256 / 2 for 32bit since DstSync is always Half)
    std::uint32_t bottom_face_offset = (unpack_dst_format == to_underlying(DataFormat::Tf32) ? 256 : 512) / 2;
    // unit_dim 1 copies 2 faces before jumping so at the moment of the jump dest RWC is
    // 2*16 (two faces) -  8 (number of rows moved by current instruction)
    // unit_dim 2 copies 4 faces before jumping so at the moment of the jump dest RWC is
    // 4*16 (four faces) - 8 (number of rows moved by current instruction)
    std::uint8_t unit_dim_1_forward_jump = bottom_face_offset - (1 * (TILE_NUM_FACES / 2) * FACE_R_DIM - 8);
    std::uint8_t unit_dim_2_forward_jump = bottom_face_offset - (2 * (TILE_NUM_FACES / 2) * FACE_R_DIM - 8);

    // jumping back to the offset for the next tile is logically -bottom_face_offset if dest RWC is at the correct offset for the bottom faces of the next tile
    // only catch is the need to compensate for the current instruction, for unit_dim 1 that is MOVA2D while for unit_dim 2 and 3 that is MOVB2D
    std::int16_t unit_dim_1_backward_jump = -bottom_face_offset + 8;
    std::int16_t unit_dim_2_backward_jump = -bottom_face_offset + 4;

    if (unit_dim == 1)
    {
        // this follows MOVA2D in src and jumps to the offset for the bottom faces (for unit_dim 1 and 2, for unit_dim 3 that is handled the other way)
        addr_mod_t {
            .srca = {.incr = 8},
            .dest = {.incr = unit_dim_1_forward_jump},
        }
            .set(ADDR_MOD_3);

        // this jumps back to the offset for the next tile, RWCs for source registers are reset separately when clearing dvalids
        addr_mod_t {
            .dest = {.incr = unit_dim_1_backward_jump},
        }
            .set(ADDR_MOD_0);
    }
    else
    {
        // this follows MOVA2D in src and jumps to the offset for the bottom faces (for unit_dim 1 and 2, for unit_dim 3 that is handled the other way)
        addr_mod_t {
            .srca = {.incr = 8},
            .dest = {.incr = unit_dim_2_forward_jump},
        }
            .set(ADDR_MOD_3);

        // this jumps back to the offset for the next tile, RWCs for source registers are reset separately when clearing dvalids
        addr_mod_t {
            .dest = {.incr = unit_dim_2_backward_jump},
        }
            .set(ADDR_MOD_0);
    }
}

inline void _llk_math_fast_tilize_mop_config_()
{
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,
        false,
        TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0),
        TT_OP_NOP,
        TT_OP_NOP);

    tmp.program();
}

inline void _llk_math_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t unit_dim)
{
    // even though MOVA2D and MOVB2D are supposed to ignore ALU_ACC_CTRL_Fp32_enabled some parts still rely on it (not sure why)
    // it would be easier if they just fully respected ALU_ACC_CTRL_Fp32_enabled but it's a hardware quirk
    // so in non Tf32 cases, clear it to fully ignore FP32 dest mode
    // not sure why it doesn't work if CFG_STATE_ID_StateID is not 1
    if (unpack_dst_format != to_underlying(DataFormat::Tf32))
    {
        TT_SETC16(CFG_STATE_ID_StateID_ADDR32, 1);
        TTI_NOP;
        TTI_NOP;
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }

    // everything else is quite standard math init
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_fast_tilize_addrmod_config_(unpack_dst_format, unit_dim);

    _llk_math_fast_tilize_mop_config_();
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_fast_tilize_uninit_(const std::uint32_t unpack_dst_format)
{
    // if ALU_ACC_CTRL_Fp32_enabled was previously cleared, restore it
    // still not sure why this CFG_STATE_ID_StateID manipulation is needed
    if (unpack_dst_format != to_underlying(DataFormat::Tf32))
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
        TT_SETC16(CFG_STATE_ID_StateID_ADDR32, 0);
        TTI_NOP;
        TTI_NOP;
    }
}

inline void _llk_math_fast_tilize_block_(
    const std::uint32_t dst_index, const std::uint32_t unpack_dst_format, const std::uint32_t unit_dim, const std::uint32_t num_units)
{
    // split dest and write the top faces in the first half and the bottom faces in the second half (or more precisely quarter, since dest sync half)
    // make life easier by lying to set_dst_write_addr that tile shape is 32x16 so correct stride is obtained for dst_index
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // srcA has the full tile, copy the top faces first
            // inside mop:
            // for (uint j = 0; j < 3; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 3 - 1, 0x0);
            // finish with the top faces and jump to the offset for the bottom faces
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
            // copy the bottom faces
            // inside mop:
            // for (uint j = 0; j < 3; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 3 - 1, 0x0);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_mova2d::MOV_8_ROWS, 0);
            // clear just srcA dvalid since it's the only one set by the unpacker for unit_dim 1 and src RWCs
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else if (unit_dim == 2)
        {
            // srcA has the top faces (4 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 7; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 7 - 1, 0x0);
            // finish with the top faces and jump to the offset for the bottom faces
            TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_3, p_mova2d::MOV_8_ROWS, 0);
            // srcB has the bottom faces (4 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 15 - 1, 0xFFFF);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
            // clear both dvalids and src RWCs
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        else if (unit_dim == 3)
        {
            // srcA has the top 8 rows of the top faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 6; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 6 - 1, 0x0);
            // srcB has the bottom 8 rows of the top faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 12; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 12 - 1, 0xFFFF);
            // done with the top faces, clear dvalids and src RWCs, next banks contain bottom faces
            // also clear dest RWC since we use dest offset for forward jump here
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
            // don't have enough address mods to have unit_dim 3 forward jump so dest offset is used here
            std::uint32_t top_face_offset = dst_index + i * 3; // copy 3 tiles per iteration
            // offset to the bottom is the number of tiles that fit into the dest bank
            // since half size faces are specified, this gets into the correct position in the second half
            std::uint32_t bottom_face_offset = top_face_offset + (unpack_dst_format == to_underlying(DataFormat::Tf32) ? 4 : 8);
            math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(bottom_face_offset);
            // srcA has the top 8 rows of the bottom faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 6; j++)
            // {
            //     TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 6 - 1, 0x0);
            // srcB has the bottom 8 rows of the bottom faces (6 of them), copy them
            // inside mop:
            // for (uint j = 0; j < 11; j++)
            // {
            //     TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, 11 - 1, 0xFFFF);
            // finish with the bottom faces and jump back to the offset for the next tile
            TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
            // clear both dvalids and src RWCs
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
    }
    math::clear_dst_reg_addr();
}
