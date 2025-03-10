// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool diagonal = false>
inline void _llk_pack_untilize_configure_addrmod_()
{
    static_assert(!diagonal, "Diagonal not supported");

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_0);

    // addr_mod_pack_t{
    //     .y_src = { .incr = 1, .clr = 0},
    // }.set(ADDR_MOD_1);

    addr_mod_pack_t {
        .z_src = {.clr = 1},
    }
        .set(ADDR_MOD_2);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim = block_ct_dim, bool diagonal = false>
inline void _llk_pack_untilize_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, bool narrow_row = false, std::uint32_t row_num_datums = TILE_C_DIM)
{
    constexpr uint MEGAROW          = 1;
    constexpr uint ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr uint MOP_INNER_LOOP   = block_ct_dim;

    // Loop until face_r_dim - 1.
    // Last row of face needs to be handled differently depending on num_faces, block_ct and full_ct.
    const uint MOP_OUTER_LOOP = face_r_dim - 1;

    const uint PACK_INTF_SEL = (narrow_row) ? p_pacr::SINGLE_INTF_ACTIVE : ((num_faces > 1) ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::SINGLE_INTF_ACTIVE);

    bool outer_loop_valid = (MOP_OUTER_LOOP > 0) && (MOP_OUTER_LOOP < 128);
    bool inner_loop_valid = (MOP_INNER_LOOP > 0) && (MOP_INNER_LOOP < 128);
    // Currently no way to check if MOP properly configured when issuing MOP instruction
    // so we check here and guard with a default configuration that only has NOPs
    if (outer_loop_valid && inner_loop_valid)
    {
        /*
        Each pack instruction does 2x16 datums if (num_faces>1)
        Each row of 16 datums, has a stride of 16 from dest read
        Dest row read in inner loop:
        tile 0: row 0, row 16
        tile 1: row 64, row 80
        .
        tile block_ct_dim-1: row 64*(block_ct_dim-1), row 64*(block_ct_dim-1)+16
        */

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                0,
                PACK_INTF_SEL,
                0,
                MEGAROW,
                p_pacr::NO_CTXT_CTRL,
                0,
                0),
            TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0) // w cnt points to the next tile
        );

        // reset ch0_w counters
        tmp.set_start_op(TT_OP_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010));

        if constexpr (block_ct_dim != full_ct_dim)
        {
            const std::uint32_t replay_buf_len = 4;
            load_replay_buf(
                ckernel::packer::replay_buf_offset,
                replay_buf_len,
                false,
                []
                {
                    // update l1 address
                    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
                    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
                    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
                    TTI_NOP;
                });

            tmp.set_end_ops(
                TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0),                             // inc ch0_y counters
                TT_OP_REPLAY(ckernel::packer::replay_buf_offset, replay_buf_len, 0, 0) // update row address
            );
        }
        else
        {
            tmp.set_end_op(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0) // inc ch0_y counters
            );
        }
        tmp.program(instrn_buffer);
    }
    // If wanted MOP config is not valid, create a default one.
    // This is due to not being able to check if MOP config is valid before issuing in runtime.
    else
    {
        ckernel::ckernel_template tmp(1, 1);
        tmp.program(instrn_buffer);
    }
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_init_(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    static_assert(!diagonal, "Diagonal not supported");
    _llk_pack_untilize_configure_addrmod_<diagonal>();

    _llk_pack_untilize_mop_config_<block_ct_dim, full_ct_dim, diagonal>(face_r_dim, num_faces, narrow_row, row_num_datums);

    // Set CH0 Zstride = 2x16x16 faces, .z_src = {.incr = 1} jumps 2 faces
    uint x_stride       = (uint)(pack_src_format & 0x3) == (uint)DataFormat::Float32 ? 4 : (uint)(pack_src_format & 0x3) == (uint)DataFormat::Float16 ? 2 : 1;
    uint y_stride       = FACE_C_DIM * x_stride;
    const uint z_stride = 2 * face_r_dim * y_stride;
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(z_stride);

    if (block_ct_dim != full_ct_dim)
    {
        const std::uint32_t output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces > 1) ? (num_faces >> 1) : 1) * FACE_C_DIM);
        TT_SETDMAREG(0, LOWER_HALFWORD(output_addr_offset / 16), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET)); // store 16B aligned row offset address
    }
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_(
    const std::uint32_t address,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim      = FACE_R_DIM,
    const std::uint32_t num_faces       = 4,
    const std::uint32_t tile_dst_offset = 0)
{
    // program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal>(address, pack_dst_format);
    program_packer_destination(address);
    const std::uint32_t num_faces_per_rdim_tile = (num_faces > 2) ? 2 : 1;
    const uint PACK_INTF_SEL = (narrow_row) ? p_pacr::SINGLE_INTF_ACTIVE : ((num_faces > 1) ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::SINGLE_INTF_ACTIVE);

    TT_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011); // reset ch0 zw counters
    TT_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011); // reset ch0 xy counters
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_dst_offset);

    for (std::uint32_t face = 0; face < num_faces_per_rdim_tile; face++)
    {
        ckernel::ckernel_template::run(instrn_buffer);

        //-----------------------------------------------------------------------
        // Handle last row of face, i.e. last outer_loop iteration of MOP.
        // Start OP is the same for all cases.
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010); // reset ch0 W counter

        // Inner loop of MOP.
        for (std::uint32_t i = 0; i < block_ct_dim; i++)
        {
            // Close block if it is the last PACR instruction of the block.
            if ((face == num_faces_per_rdim_tile - 1) && (i == block_ct_dim - 1) && (block_ct_dim == full_ct_dim))
            {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_STRIDED_MODE,
                    ADDR_MOD_2,
                    p_pacr::ADDR_CNT_CTXT_0,
                    p_pacr::P_ZERO_OUTPUT_DISABLED,
                    PACK_INTF_SEL,
                    0,
                    0 /*MEGAROW*/,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    1);
            }
            else
            {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_STRIDED_MODE,
                    ADDR_MOD_0,
                    p_pacr::ADDR_CNT_CTXT_0,
                    p_pacr::P_ZERO_OUTPUT_DISABLED,
                    PACK_INTF_SEL,
                    0,
                    1 /*MEGAROW*/,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
            }

            TTI_INCADCZW(p_setadc::PAC, 0, 0, 1, 0); // w cnt points to the next tile
        }

        // End OP.
        TTI_INCADCXY(p_setadc::PAC, 0, 0, 1, 0); // inc ch0_y counters
        if (block_ct_dim != full_ct_dim)
        {
            // update l1 address
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
            TTI_NOP;
        }
        //-----------------------------------------------------------------------

        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);         // z cnt increments by 2xface_r_dimxFACE_C_DIM
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010); // reset ch0_y counters
    }

    TT_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); // reset z counters
}
