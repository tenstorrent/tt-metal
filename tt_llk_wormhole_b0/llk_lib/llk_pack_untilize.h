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
#include "lltt.h"
#include "sfpi.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool diagonal = false, bool narrow_row = false>
inline void _llk_pack_untilize_configure_addrmod_()
{
    if constexpr (diagonal || narrow_row)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 1},
        }
            .set(ADDR_MOD_0);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        }
            .set(ADDR_MOD_0);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_2);

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const uint PACKCNT              = diagonal ? (num_faces > 2 ? num_faces / 2 : num_faces) : ((face_r_dim < FACE_R_DIM) ? 1 : num_faces);
    constexpr uint MEGAROW          = 1;
    constexpr uint ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr uint MOP_INNER_LOOP   = narrow_row ? (TILE_R_DIM / 4) : (diagonal ? FACE_R_DIM - 1 : 1);
    constexpr uint MOP_OUTER_LOOP   = narrow_row ? 1 : block_ct_dim;

    if constexpr (diagonal)
    {
        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_INCADCXY(p_setadc::PAC, 0, 1, 0, 1),
            TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_end_ops(TT_OP_SETADCXX(p_setadc::PAC, 1 - 1, 0x0), TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
        tmp.program(instrn_buffer);
    }
    else
    {
        if constexpr (narrow_row)
        { // always
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
            tmp.program(instrn_buffer);
        }
        else
        {
            if (num_faces > 1)
            {
                // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
                ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
                tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
                tmp.set_end_ops(
                    TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0),
                    TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
                tmp.program(instrn_buffer);
            }
            else
            {
                ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
                tmp.set_end_op(TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0)); // w cnt points to the next tile
                tmp.program(instrn_buffer);
            }
        }

        if (block_ct_dim != full_ct_dim)
        {
            const std::uint32_t replay_buf_len = 10;
            lltt::record(ckernel::packer::replay_buf_offset, replay_buf_len);
            TTI_PACR(ADDR_MOD_3, 0, 0xf, 0, 0, 1, 1); // close block
            // update l1 address
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 1, p_gpr_pack::OUTPUT_ADDR + 1, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 2, p_gpr_pack::OUTPUT_ADDR + 2, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR + 3, p_gpr_pack::OUTPUT_ADDR + 3, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 3);
            TTI_NOP;
        }
    }
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
inline void _llk_pack_untilize_init_(const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    _llk_pack_untilize_configure_addrmod_<diagonal, narrow_row>();

    _llk_pack_untilize_mop_config_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(face_r_dim, num_faces);

    if (block_ct_dim != full_ct_dim)
    {
        const std::uint32_t output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces > 1) ? num_faces / 2 : 1) * FACE_C_DIM);
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
    const std::uint32_t num_faces       = 4 /*not used*/,
    const std::uint32_t tile_dst_offset = 0)
{
    program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal, row_num_datums>(address, pack_dst_format);

    if constexpr (narrow_row)
    {
        ckernel::ckernel_template::run(instrn_buffer);
    }
    else
    {
        const std::uint32_t num_rows = (face_r_dim < FACE_R_DIM) ? face_r_dim : TILE_R_DIM / 4;

        for (std::uint32_t row = 0; row < num_rows; row++)
        {
            TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_dst_offset); // Clear tile counter
            ckernel::ckernel_template::run(instrn_buffer);
            TTI_ADDRCRXY(p_setadc::PAC, 0, 0, 1, 0, 0b0010); // Read new row in the tile
            if constexpr (block_ct_dim != full_ct_dim)
            {
                lltt::replay(ckernel::packer::replay_buf_offset, 10); // update row address
            }
        }
    }

    if constexpr (block_ct_dim == full_ct_dim)
    {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close block
    }
}
