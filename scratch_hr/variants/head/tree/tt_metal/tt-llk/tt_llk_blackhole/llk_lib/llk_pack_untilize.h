// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Configure the ADDR_MOD slots used by the untilize pack MOP.
 *
 * ADDR_MOD_0 (every PACR of a row but the last) steps Z to the next tile in Dest. The row-closing PACR moves
 * y_src to the next face-row and clears Z back to the block's first tile; ADDR_MOD_1 also steps the channel 1
 * Y counter that selects the L1 output row (rows that end the L1 stream), ADDR_MOD_2 does not (rows that
 * continue it).
 */
inline void _llk_pack_untilize_configure_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0},
        .z_src = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t {
        .y_src = {.incr = 1, .clr = 0},
        .y_dst = {.incr = 1, .clr = 0},
        .z_src = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_pack_t {
        .y_src = {.incr = 1, .clr = 0},
        .z_src = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);
}

// ADDR_MOD_3 also steps channel 1 Z by one tile row into L1, which ADDR_MOD_1 clears again.
inline void _llk_pack_untilize_configure_first_tile_stream_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 1, .clr = 0},
        .y_dst = {.incr = 1, .clr = 0},
        .z_src = {.incr = 0, .clr = 1},
        .z_dst = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_pack_t {
        .z_src = {.incr = 1, .clr = 0},
        .z_dst = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_3);
}

/*
block_ct_dim represents the number of input tiles in a block.
dense is used with num_faces == 2 and even block_ct_dim, where two 16x32 (or smaller) tiles are packed in a single 32x32 tile region in dest.
*/
/**
 * @brief Build and program the packer MOP template for an untilize (tilized -> row-major) pack.
 *
 * Programs a MOP that walks face rows in the outer loop and tiles within the block in the inner loop,
 * using DST_STRIDED_MODE so each PACR packs a row from each tile. The PACR address modes step the tile (Z),
 * the face-row (Y) and the L1 output row (channel 1 Y), so a row is block_ct_dim PACRs, plus the CFGSHIFTMASK row
 * step when l1_row_step_by_cfg and the fillers when paced.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam narrow_row: True when faces occupy only the first column of the tile (single packer interface).
 * @tparam dense: True to pack two tiles into one 32x32 dest region using all interfaces; requires num_faces == 2 and even block_ct_dim.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param row_ends_stream: True to close every row with Last (rows not contiguous in L1, or 32-bit Dest reads of rows of whole 16-byte units).
 * @param l1_row_step_by_cfg: True to advance the L1 destination address per row with CFGSHIFTMASK, for row strides
 *        the channel 1 Y stride field cannot hold.
 * @param first_tile_stream: True to write the first tile of every row as its own L1 stream (Last), so that the
 *        64-datum flush of an 8-bit output lands on the next tile, which the rest of the row then rewrites.
 * @note @ref _llk_pack_untilize_configure_addrmod_ must have programmed the ADDR_MOD slots.
 */
template <std::uint32_t block_ct_dim, bool narrow_row = false, bool dense = false, bool pace = false>
inline void _llk_pack_untilize_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool row_ends_stream     = true,
    const bool l1_row_step_by_cfg  = false,
    const bool first_tile_stream   = false)
{
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    LLK_ASSERT(!first_tile_stream || (!pace && !dense && block_ct_dim > 1), "first_tile_stream needs a plain row of two or more tiles");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");
    /*
    Outer loop iterates over the rows in the block, while the inner loop iterates
    over each tile in the block.
    When dense, we use all 4 interfaces to pack out a row each from 4 faces (2 tiles) that end up contiguous in L1
    because offsets align well and it improves perf, thus we halve the number of mop inner loops.
    */
    constexpr std::uint32_t MOP_INNER_LOOP = dense ? block_ct_dim / 2 : block_ct_dim;
    const std::uint32_t MOP_OUTER_LOOP     = face_r_dim;

    // For narrow row, the faces are stored in the first column of the tile, therefore requiring only one packer interface.
    const std::uint32_t PACK_INTF_SEL = (dense)                          ? p_pacr::ALL_INTF_ACTIVE
                                        : (narrow_row || num_faces == 1) ? p_pacr::SINGLE_INTF_ACTIVE
                                                                         : p_pacr::TWO_INTFS_ACTIVE;
    /*
    When using DST_STRIDED_MODE, each packer interface has a stride of 16*block_size,
    where block_size is set to be the size of a row within face.
    Each PACR instruction packs 2x16 datums if (num_faces>1), meaning that it would
    pack out one row for each tile in the block.
    In the inner loop, for each tile, the rows that get packed from dest register
    in the first outer loop iteration are:
    tile 0: row 0, row 16
    tile 1: row 64, row 80
    tile block_ct_dim-1: row 64*(block_ct_dim-1), row 64*(block_ct_dim-1)+16
    This processes is repeated for each row of the block in dest.
    */
    const std::uint32_t pacr_op = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_0,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        PACK_INTF_SEL,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        0);
    auto program_mop = [&](ckernel::ckernel_template& tmp)
    {
        // A PACR with Last makes the next one start at a fresh L1 address.
        const std::uint32_t row_close_addr_mod = row_ends_stream ? ADDR_MOD_1 : ADDR_MOD_2;
        const std::uint32_t row_close_op       = TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            row_close_addr_mod,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            row_ends_stream ? 1 : 0);
        const std::uint32_t pass_close_op = TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            row_close_addr_mod,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            PACK_INTF_SEL,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1);

        tmp.set_last_inner_loop_instr(row_close_op);
        tmp.set_last_outer_loop_instr(pass_close_op);

        if (l1_row_step_by_cfg)
        {
            const std::uint32_t replay_buf_len = 2;
            load_replay_buf(
                ckernel::packer::replay_buf_offset,
                replay_buf_len,
                []
                {
                    TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
                    TTI_NOP;
                });
            tmp.set_end_op(lltt::replay_insn(ckernel::packer::replay_buf_offset, replay_buf_len));
        }
        else if constexpr (pace)
        {
            tmp.set_end_ops(TT_OP_DMANOP, TT_OP_DMANOP);
        }
        if constexpr (pace)
        {
            tmp.set_start_op(TT_OP_DMANOP);
        }
        if (first_tile_stream)
        {
            tmp.set_start_op(TT_OP_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_3,
                p_pacr::ADDR_CNT_CTXT_0,
                0,
                PACK_INTF_SEL,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                1));
        }

        tmp.program();
    };
    if constexpr (pace)
    {
        ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_DMANOP, pacr_op);
        program_mop(tmp);
    }
    else
    {
        ckernel::ckernel_template tmp(MOP_OUTER_LOOP, first_tile_stream ? MOP_INNER_LOOP - 1 : MOP_INNER_LOOP, pacr_op);
        program_mop(tmp);
    }
}

// A packer interface reads x_end + 1 consecutive datums, so each tile row is the left face's 16 datums on interface 0, then the rest
// from the right face on interface 1 with x_end lowered, in one L1 stream. ADDR_MOD_3 must hold no increments, x_end FACE_C_DIM - 1 on entry.
template <std::uint32_t block_ct_dim, std::uint32_t row_num_datums>
inline void _llk_pack_untilize_split_row_mop_config_(const std::uint32_t face_r_dim, const bool row_ends_stream, const bool l1_row_step_by_cfg)
{
    static_assert((row_num_datums > FACE_C_DIM) && (row_num_datums < TILE_C_DIM), "a split row is wider than a face and narrower than a tile");
    constexpr std::uint32_t seq_len        = 4;
    constexpr std::uint32_t seq_start      = ckernel::packer::replay_buf_offset + 2;
    constexpr std::uint32_t right_intf     = 0b0010;
    const std::uint32_t row_close_addr_mod = row_ends_stream ? ADDR_MOD_1 : ADDR_MOD_2;

    load_replay_buf(
        seq_start,
        3 * seq_len,
        [row_close_addr_mod, row_ends_stream]
        {
            const std::uint32_t right_addr_mod[3] = {ADDR_MOD_0, row_close_addr_mod, row_close_addr_mod};
            const std::uint32_t right_last[3]     = {0, row_ends_stream ? 1u : 0u, 1};
            for (std::uint32_t i = 0; i < 3; i++)
            {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_STRIDED_MODE,
                    ADDR_MOD_3,
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    p_pacr::SINGLE_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
                TTI_SETADCXX(p_setadc::PAC, row_num_datums - FACE_C_DIM - 1, 0x0);
                TT_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_STRIDED_MODE,
                    right_addr_mod[i],
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    right_intf,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    right_last[i]);
                TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
            }
        });

    ckernel::ckernel_template tmp(face_r_dim, block_ct_dim, lltt::replay_insn(seq_start, seq_len));
    tmp.set_last_inner_loop_instr(lltt::replay_insn(seq_start + seq_len, seq_len));
    tmp.set_last_outer_loop_instr(lltt::replay_insn(seq_start + 2 * seq_len, seq_len));
    if (l1_row_step_by_cfg)
    {
        load_replay_buf(
            ckernel::packer::replay_buf_offset,
            2,
            []
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
                TTI_NOP;
            });
        tmp.set_end_op(lltt::replay_insn(ckernel::packer::replay_buf_offset, 2));
    }
    tmp.program();
}

// Rows of a 32-bit read into a 16-bit output in blocks of three keep the counter form: a W step (INCADCZW) before every
// PACR and a CFGSHIFTMASK row step. The address-mode form pays a start-up cost there that pacing does not remove.
inline void _llk_pack_untilize_counter_form_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t {
        .y_src = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_1);
}

inline void _llk_pack_untilize_counter_form_mop_config_(const std::uint32_t block_ct_dim, const std::uint32_t face_r_dim)
{
    ckernel::ckernel_template tmp(
        face_r_dim,
        block_ct_dim,
        TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::TWO_INTFS_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0));
    // W back to the block's first tile (the W_Cr the execute sets) at the start of every row.
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010));

    const std::uint32_t replay_buf_len = 2;
    load_replay_buf(
        ckernel::packer::replay_buf_offset,
        replay_buf_len,
        []
        {
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
            TTI_NOP;
        });
    tmp.set_end_op(lltt::replay_insn(ckernel::packer::replay_buf_offset, replay_buf_len));

    const std::uint32_t row_close_op = TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    tmp.set_last_inner_loop_instr(row_close_op);
    tmp.set_last_outer_loop_instr(row_close_op);
    tmp.program();
}

// _llk_pack_untilize_init_ for a caller that names datum sizes or turns the pace off: known sizes fold the run-time
// choices as compile-time formats do.
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense,
    bool pace_32bit_rows,
    std::uint32_t src_datum_size,
    std::uint32_t dst_datum_size>
inline void _llk_pack_untilize_init_sized_(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    static_assert(block_ct_dim <= (dense ? 16 : 8), "block_ct_dim must be <= 8 when not dense, <= 16 when dense");
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    static_assert(block_ct_dim <= full_ct_dim, "a block cannot be wider than its row");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");
    LLK_ASSERT(num_faces < 4 || face_r_dim == FACE_R_DIM, "four faces need full face rows");
    // Each row of a one-tile block is its own 32-datum stream, and an Fp8_e4m3 stream pads to 64 datums.
    LLK_ASSERT(
        !(block_ct_dim == 1 && full_ct_dim > 1 && !narrow_row && static_cast<DataFormat>(pack_dst_format) == DataFormat::Fp8_e4m3),
        "an Fp8_e4m3 output needs blocks of two or more tiles when a row has several blocks");

    if constexpr (narrow_row)
    {
        // Changed to check against TILE_C_DIM instead of FACE_C_DIM until tt-metal#24095 is investigated.
        static_assert(row_num_datums < TILE_C_DIM, "row_num_datums must be set to less than TILE_C_DIM for narrow_row packing");
    }
    LLK_ASSERT(src_datum_size == 0 || src_datum_size == datum_size_in_bytes(pack_src_format), "src_datum_size must match pack_src_format");
    LLK_ASSERT(
        dst_datum_size == 0 || (dst_datum_size == SCALE_DATUM_SIZE(pack_dst_format, 1) && (dst_datum_size == 1) == IS_8BIT_FORMAT(pack_dst_format)),
        "dst_datum_size must match pack_dst_format");
    const std::uint32_t src_size = (src_datum_size != 0) ? src_datum_size : datum_size_in_bytes(pack_src_format);
    const bool dst_8bit          = (dst_datum_size != 0) ? (dst_datum_size == 1) : IS_8BIT_FORMAT(pack_dst_format);
    // Only a caller naming both sizes reaches this: kernels read a 16-bit Dest for 16-bit outputs.
    constexpr bool src32_dst16 = (src_datum_size == 4) && (dst_datum_size == 2) && !narrow_row && !dense;
    if constexpr (src32_dst16 && (block_ct_dim == 3) && (full_ct_dim != block_ct_dim))
    {
        LLK_ASSERT(num_faces == 4, "the counter form packs four-face tiles");
        _llk_pack_untilize_counter_form_addrmod_();
        _llk_pack_untilize_counter_form_mop_config_(block_ct_dim, face_r_dim);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(2 * face_r_dim * FACE_C_DIM * src_datum_size);
        const std::uint32_t row_offset_16B = dst_datum_size * full_ct_dim * 2 * FACE_C_DIM / 16;
        TT_SETDMAREG(0, LOWER_HALFWORD(row_offset_16B), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        TT_SETDMAREG(0, UPPER_HALFWORD(row_offset_16B), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR_OFFSET, 0, SCRATCH_SEC2_val_ADDR32);
        TTI_NOP;
        TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
        return;
    }

    std::uint32_t output_addr_offset;
    if constexpr (narrow_row)
    {
        output_addr_offset =
            (dst_datum_size != 0) ? dst_datum_size * full_ct_dim * row_num_datums : SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * row_num_datums);
    }
    else
    {
        output_addr_offset = (dst_datum_size != 0) ? dst_datum_size * full_ct_dim * ((num_faces == 1) ? 1 : 2) * FACE_C_DIM
                                                   : SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces == 1) ? 1 : 2) * FACE_C_DIM);
    }
    // A full-width block is one L1 stream per face pair; 32-bit reads close every row to leave Dest cycles to
    // an unpacker writing the other half.
    constexpr bool l1_rows_contiguous = (full_ct_dim == block_ct_dim);
    // An 8-bit output reaches L1 in 64-datum units: a flush pads a shorter stream with zeros up to the unit.
    // A stream starts on a 16-byte L1 boundary, so a narrow row of another byte size stays in one stream.
    const bool src_32b            = src_size == 4;
    const bool unaligned_narrow   = narrow_row && (output_addr_offset % 16 != 0);
    const bool row_ends_stream    = !l1_rows_contiguous || (src_32b && !dst_8bit && !src32_dst16 && !unaligned_narrow);
    constexpr bool odd_block_form = !l1_rows_contiguous && (block_ct_dim % 2 == 1) && (block_ct_dim > 1) && !narrow_row && !dense;
    const bool first_tile_stream  = odd_block_form && dst_8bit;
    constexpr bool split_row      = narrow_row && (row_num_datums > FACE_C_DIM);
    LLK_ASSERT(!split_row || (num_faces > 1), "a narrow row wider than a face needs the right faces");
    // The channel 1 Y stride field is 16 bits, and the packer keeps the channel 1 offset only within 256 KiB.
    const std::uint32_t rows_per_call = face_r_dim * ((num_faces > 2) ? 2 : 1);
    // 32-bit rows of three tiles are paced, and in the block form step L1 by CFGSHIFTMASK: packed back to back they
    // cost an unpacker writing Dest more L1 cycles than they save.
    constexpr bool may_pace = ((block_ct_dim == 3) || (src32_dst16 && (block_ct_dim == 4) && !l1_rows_contiguous)) && !split_row;
    const bool pace         = pace_32bit_rows && may_pace && src_32b && !dst_8bit;
    const bool l1_row_step_by_cfg =
        row_ends_stream &&
        ((pace && !l1_rows_contiguous) || (output_addr_offset > (PCK0_ADDR_CTRL_XY_REG_1_Ystride_MASK >> PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT)) ||
         (rows_per_call * output_addr_offset > 256 * 1024));

    _llk_pack_untilize_configure_addrmod_();

    if constexpr (split_row)
    {
        addr_mod_pack_t {}.set(ADDR_MOD_3);
        _llk_pack_untilize_split_row_mop_config_<block_ct_dim, row_num_datums>(face_r_dim, row_ends_stream, l1_row_step_by_cfg);
    }
    else if constexpr (may_pace)
    {
        if (pace)
        {
            _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense, true>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg);
        }
        else
        {
            _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg, first_tile_stream);
        }
    }
    else
    {
        _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg, first_tile_stream);
    }
    if constexpr (odd_block_form)
    {
        if (first_tile_stream)
        {
            _llk_pack_untilize_configure_first_tile_stream_addrmod_();
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(
                (dst_datum_size != 0) ? dst_datum_size * TILE_C_DIM : SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM));
        }
    }

    const std::uint32_t z_stride = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM * src_size;
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(z_stride);

    // Each call loads this word into PCK0_ADDR_CTRL_XY_REG_1 and clears the register again, so no later pack sees it.
    const std::uint32_t ch1_y_stride = l1_row_step_by_cfg ? 0 : (output_addr_offset << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT);
    TT_SETDMAREG(0, LOWER_HALFWORD(ch1_y_stride), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(ch1_y_stride), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    if (l1_row_step_by_cfg)
    {
        TT_SETDMAREG(0, LOWER_HALFWORD(output_addr_offset / 16), 0, LO_16(p_gpr_pack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(output_addr_offset / 16), 0, HI_16(p_gpr_pack::TMP0));
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    if (l1_row_step_by_cfg)
    {
        TTI_WRCFG(p_gpr_pack::TMP0, 0, SCRATCH_SEC2_val_ADDR32);
        TTI_NOP;
    }

    // Always include setup calls for safety (as recommended by maintainer)
    // Program packer to pack out the correct number of datums per row
    if constexpr (narrow_row)
    {
        TTI_SETADCXX(p_setadc::PAC, (split_row ? FACE_C_DIM : row_num_datums) - 1, 0x0);
    }
    else
    {
        TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    }
}

/**
 * @brief Initialize the packer for an untilize pack op.
 *
 * Keeps the channel 1 Y stride (one L1 output row) in a GPR for the execute calls.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Number of tiles in the output row; a row may be split into blocks of different widths.
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set.
 * @tparam dense: True to pack two tiles into one dest region; requires num_faces == 2 and even block_ct_dim.
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @tparam pace_32bit_rows: False to leave 32-bit rows in blocks of three unpaced: for a Dest that math fills, or a kernel
 *         that reads equal or faster without the pace.
 * @tparam src_datum_size, dst_datum_size: Bytes per datum of pack_src_format and pack_dst_format when the caller knows them
 *         at compile time but passes the formats at run time, 0 otherwise (see _llk_pack_untilize_init_sized_).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note On the math thread, @ref _llk_math_eltwise_unary_datacopy_ (A2D) populates the dest register this packer reads.
 * @note Pair with @ref _llk_pack_untilize_uninit_ after the matching @ref _llk_pack_untilize_ execute calls.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense                   = false,
    bool pace_32bit_rows      = true,
    std::uint32_t src_datum_size = 0,
    std::uint32_t dst_datum_size = 0>
inline void _llk_pack_untilize_init_(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    if constexpr (!pace_32bit_rows || (src_datum_size != 0) || (dst_datum_size != 0))
    {
        _llk_pack_untilize_init_sized_<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense, pace_32bit_rows, src_datum_size, dst_datum_size>(
            pack_src_format, pack_dst_format, face_r_dim, num_faces);
        return;
    }

    static_assert(block_ct_dim <= (dense ? 16 : 8), "block_ct_dim must be <= 8 when not dense, <= 16 when dense");
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    static_assert(block_ct_dim <= full_ct_dim, "a block cannot be wider than its row");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");
    LLK_ASSERT(num_faces < 4 || face_r_dim == FACE_R_DIM, "four faces need full face rows");
    // Each row of a one-tile block is its own 32-datum stream, and an Fp8_e4m3 stream pads to 64 datums.
    LLK_ASSERT(
        !(block_ct_dim == 1 && full_ct_dim > 1 && !narrow_row && static_cast<DataFormat>(pack_dst_format) == DataFormat::Fp8_e4m3),
        "an Fp8_e4m3 output needs blocks of two or more tiles when a row has several blocks");

    if constexpr (narrow_row)
    {
        // Changed to check against TILE_C_DIM instead of FACE_C_DIM until tt-metal#24095 is investigated.
        static_assert(row_num_datums < TILE_C_DIM, "row_num_datums must be set to less than TILE_C_DIM for narrow_row packing");
    }

    std::uint32_t output_addr_offset;
    if constexpr (narrow_row)
    {
        output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * row_num_datums);
    }
    else
    {
        output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces == 1) ? 1 : 2) * FACE_C_DIM);
    }
    // A full-width block is one L1 stream per face pair; 32-bit reads close every row to leave Dest cycles to
    // an unpacker writing the other half.
    constexpr bool l1_rows_contiguous = (full_ct_dim == block_ct_dim);
    // An 8-bit output reaches L1 in 64-datum units: a flush pads a shorter stream with zeros up to the unit.
    // A stream starts on a 16-byte L1 boundary, so a narrow row of another byte size stays in one stream.
    const bool src_32b            = datum_size_in_bytes(pack_src_format) == 4;
    const bool unaligned_narrow   = narrow_row && (output_addr_offset % 16 != 0);
    const bool row_ends_stream    = !l1_rows_contiguous || (src_32b && !IS_8BIT_FORMAT(pack_dst_format) && !unaligned_narrow);
    constexpr bool odd_block_form = !l1_rows_contiguous && (block_ct_dim % 2 == 1) && (block_ct_dim > 1) && !narrow_row && !dense;
    const bool first_tile_stream  = odd_block_form && IS_8BIT_FORMAT(pack_dst_format);
    constexpr bool split_row      = narrow_row && (row_num_datums > FACE_C_DIM);
    LLK_ASSERT(!split_row || (num_faces > 1), "a narrow row wider than a face needs the right faces");
    // The channel 1 Y stride field is 16 bits, and the packer keeps the channel 1 offset only within 256 KiB.
    const std::uint32_t rows_per_call = face_r_dim * ((num_faces > 2) ? 2 : 1);
    // 32-bit rows of three tiles are paced, and in the block form step L1 by CFGSHIFTMASK: packed back to back they
    // cost an unpacker writing Dest more L1 cycles than they save.
    const bool pace = src_32b && (block_ct_dim == 3) && !split_row && !IS_8BIT_FORMAT(pack_dst_format);
    const bool l1_row_step_by_cfg =
        row_ends_stream &&
        ((pace && !l1_rows_contiguous) || (output_addr_offset > (PCK0_ADDR_CTRL_XY_REG_1_Ystride_MASK >> PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT)) ||
         (rows_per_call * output_addr_offset > 256 * 1024));

    _llk_pack_untilize_configure_addrmod_();

    if constexpr (split_row)
    {
        addr_mod_pack_t {}.set(ADDR_MOD_3);
        _llk_pack_untilize_split_row_mop_config_<block_ct_dim, row_num_datums>(face_r_dim, row_ends_stream, l1_row_step_by_cfg);
    }
    else if constexpr (block_ct_dim == 3)
    {
        if (pace)
        {
            _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense, true>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg);
        }
        else
        {
            _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg, first_tile_stream);
        }
    }
    else
    {
        _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense>(face_r_dim, num_faces, row_ends_stream, l1_row_step_by_cfg, first_tile_stream);
    }
    if constexpr (odd_block_form)
    {
        if (first_tile_stream)
        {
            _llk_pack_untilize_configure_first_tile_stream_addrmod_();
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM));
        }
    }

    const std::uint32_t z_stride = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM * datum_size_in_bytes(pack_src_format);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(z_stride);

    // Each call loads this word into PCK0_ADDR_CTRL_XY_REG_1 and clears the register again, so no later pack sees it.
    const std::uint32_t ch1_y_stride = l1_row_step_by_cfg ? 0 : (output_addr_offset << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT);
    TT_SETDMAREG(0, LOWER_HALFWORD(ch1_y_stride), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(ch1_y_stride), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    if (l1_row_step_by_cfg)
    {
        TT_SETDMAREG(0, LOWER_HALFWORD(output_addr_offset / 16), 0, LO_16(p_gpr_pack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(output_addr_offset / 16), 0, HI_16(p_gpr_pack::TMP0));
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    if (l1_row_step_by_cfg)
    {
        TTI_WRCFG(p_gpr_pack::TMP0, 0, SCRATCH_SEC2_val_ADDR32);
        TTI_NOP;
    }

    // Always include setup calls for safety (as recommended by maintainer)
    // Program packer to pack out the correct number of datums per row
    if constexpr (narrow_row)
    {
        TTI_SETADCXX(p_setadc::PAC, (split_row ? FACE_C_DIM : row_num_datums) - 1, 0x0);
    }
    else
    {
        TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    }
}

/**
 * @brief Untilize-pack one block of tiles from the destination register to L1.
 *
 * W selects the block's first tile in Dest; the MOP runs once per face group, with Y moved to the bottom faces and the L1 row to 16 in between.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam tile_dst_ct_offset: Compile-time column-tile offset into the destination register.
 * @tparam dense: True to pack two tiles into one dest region; requires num_faces == 2 and even block_ct_dim.
 * @param address: L1 destination base address for the block.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @tparam src_datum_size, dst_datum_size: As for @ref _llk_pack_untilize_init_, and the same values.
 * @param tile_dst_rt_offset: Runtime row-tile offset into the destination register.
 * @note Call @ref _llk_pack_untilize_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_untilize_uninit_ once all untilize-pack calls are complete.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool narrow_row                  = false,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense                       = false,
    std::uint32_t src_datum_size     = 0,
    std::uint32_t dst_datum_size     = 0>
inline void _llk_pack_untilize_(const std::uint32_t address, const std::uint32_t num_faces = 4, const std::uint32_t tile_dst_rt_offset = 0)
{
    static_assert(block_ct_dim <= (dense ? 16 : 8), "block_ct_dim must be <= 8 when not dense, <= 16 when dense");
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    static_assert(block_ct_dim <= full_ct_dim, "a block cannot be wider than its row");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");

    if constexpr ((src_datum_size == 4) && (dst_datum_size == 2) && (block_ct_dim == 3) && (full_ct_dim != block_ct_dim) && !narrow_row && !dense)
    {
        // The counter form of _llk_pack_untilize_init_sized_: W starts one tile early, as its first INCADCZW steps it.
        program_packer_destination(address);
        const std::uint32_t tile_dst_offset = tile_dst_ct_offset + tile_dst_rt_offset;
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, (15 + tile_dst_offset) & 0xF);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        for (std::uint32_t face = 0; face < 2; face++)
        {
            ckernel::ckernel_template::run();
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
            TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
        }
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
        set_dst_write_addr(tile_dst_offset);
        return;
    }

    /*
    full_ct_dim represents the number of input tiles.
    For input widths greater than 8 tiles, input is split into blocks, each block the size of block_ct_dim;
    a row may also be split into blocks of different widths. This function is called for each block.
    */
    // program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal>(address, pack_dst_format);
    program_packer_destination(address);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    const std::uint32_t num_faces_per_rdim_tile = (num_faces > 2) ? 2 : 1;

    const std::uint32_t tile_dst_offset = tile_dst_ct_offset + tile_dst_rt_offset;
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_dst_offset);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1011);

    ckernel::ckernel_template::run();
    if (num_faces_per_rdim_tile > 1)
    {
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 2 * FACE_R_DIM);
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_1, p_setadc::SET_Y, FACE_R_DIM);
        ckernel::ckernel_template::run();
    }

    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1010);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
}

/**
 * Reprograms the channel 0 Z stride to its single-face default and zeroes the channel 1 strides, as set_packer_strides leaves them.
 *
 * @param pack_src_format: Source (dest register) data format used to size the default Z stride.
 * @note Pairs with @ref _llk_pack_untilize_init_.
 */
inline void _llk_pack_untilize_uninit_(const std::uint32_t pack_src_format)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    const std::uint32_t z_stride = SCALE_DATUM_SIZE(pack_src_format, FACE_R_DIM * FACE_C_DIM);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(z_stride);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}
