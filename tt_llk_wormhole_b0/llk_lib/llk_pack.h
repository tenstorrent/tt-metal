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

template <bool untilize = false>
inline void _llk_pack_configure_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    if constexpr (untilize)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 1, .clr = 0, .cr = 1},
            .y_dst = {.incr = 1, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 0},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_1);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true>
inline void _llk_pack_mop_config_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false)
{
    static_assert(FaceLayout == DstTileFaceLayout::RowMajor, "FaceLayout must be RowMajor");

    const uint PACKCNT              = (partial_face && IS_BFP_FORMAT(pack_dst_format)) ? 1 : num_faces;
    constexpr uint MEGAROW          = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr uint MOP_INNER_LOOP   = 1;

    if constexpr (!untilize)
    {
        constexpr uint MOP_OUTER_LOOP = 1;

        ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));

        if (partial_face && IS_BFP_FORMAT(pack_dst_format))
        {
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)); // Don't close the tile, point to the next face
            tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));                                     // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            tmp.set_loop_op1(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1)); // Close the tile
        }
        // Write header to l1
        if constexpr (write_tile_header)
        {
            tmp.set_end_op(TT_OP_STOREIND(1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));
        }

        tmp.program(instrn_buffer);
    }
    else
    {
        const uint MOP_OUTER_LOOP = ((face_r_dim == 1) || narrow_tile) ? 1 : (face_r_dim >> 1);

        if ((face_r_dim == 1) || narrow_tile)
        {
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program(instrn_buffer);
        }
        else
        {
            // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.program(instrn_buffer);
        }
    }
}

template <
    bool is_fp32_dest_acc_en,
    bool is_tile_dim_reconfig_en = false,
    DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor,
    bool write_tile_header       = true>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false)
{
    reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, face_r_dim);

    if constexpr (is_tile_dim_reconfig_en)
    {
        _llk_pack_mop_config_<false, false, FaceLayout, write_tile_header>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile);
    }
}

template <bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0)
{
    configure_pack<is_fp32_dest_acc_en, untilize>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile, relu_config);
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_reduce_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0)
{
    configure_pack<is_fp32_dest_acc_en, untilize>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile, relu_config);

    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    ckernel::packer::pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask                             = 0x0;
    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 1] = 0x0001;
        if constexpr (untilize)
        {
            pack_edge_offset.f.tile_row_set_select_pack0 = 1;
            pack_edge_offset.f.tile_row_set_select_pack1 = 1;
            pack_edge_offset.f.tile_row_set_select_pack2 = 1;
            pack_edge_offset.f.tile_row_set_select_pack3 = 1;
            if (narrow_tile)
            {
                cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x55555555; // each packer packs 1x16 row
            }
            else
            {
                cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x11111111; // each packer packs 1x32 row
            }
        }
        else
        {
            pack_edge_offset.f.tile_row_set_select_pack0 = 1;
            if (narrow_tile)
            {
                pack_edge_offset.f.tile_row_set_select_pack1 = 1;
            }
            else
            {
                pack_edge_offset.f.tile_row_set_select_pack2 = 1;
            }
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x55555555; // each packer packs 1x16 row
        }
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 0] = pack_edge_offset.val;
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        pack_edge_offset.f.tile_row_set_select_pack0         = 1;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 0]            = pack_edge_offset.val;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 1]            = 0x0001;
        cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000001;
    }
    else
    {
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 0]    = pack_edge_offset.val;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + 1]    = 0xffff;

        if constexpr (untilize)
        {
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000005; // Each packer packs 1x32 row
        }
        else
        {
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000001;
        }
    }
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true>
inline void _llk_pack_init_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false)
{
    _llk_pack_configure_addrmod_<untilize>();

    _llk_pack_mop_config_<untilize, zero_output, FaceLayout, write_tile_header>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);

    program_packer_destination(address);

    mop_run(1, 1);

    if constexpr (untilize)
    {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close tile
    }
}

#include "llk_pack_untilize.h"

/*************************************************************************
 * LLK PACK FAST TILIZE (Tilize single input using both unpackers and packer)
 * unit_dim is the number of tiles processed in a single iteration, num_units is the number of units processed in a single call
 * unit_dim and num_units must match the ones given to the unpacker and math (all unit_dim usage notes from the unpacker also apply here)
 * tile_index is the index of the tile inside the destination register to read from
 * address is the 16B address of where to start packing to (usually the start of the tile row)
 * currently supports only 4 16x16 faces per tile
 * supported output formats are: FP32, FP16_B, BFP8_B
 * both dest modes are supported (same usage notes from math apply here)
 * only DstSync::SyncHalf is supported
 * tiles are expected to be split into top and bottom faces in separate halves of the active dest bank
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_tilize_hw_configure_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    configure_pack<is_fp32_dest_acc_en, false>(pack_src_format, pack_dst_format);
}

inline void _llk_pack_fast_tilize_addrmod_config_(const std::uint32_t unit_dim)
{
    // first two address mods move to the next row, the stride depends on the number of contiguous faces loaded in the single unpacker instruction
    // for unit_dim 1, that is 2 so the stride is 2, and analogously for unit_dims 2 and 3 its 4 and 6
    addr_mod_pack_t {
        .y_src = {.incr = (uint8_t)(unit_dim == 1 ? 2 : 4)},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t {
        .y_src = {.incr = 6},
    }
        .set(ADDR_MOD_2);

    // this address mod moves to the same face in the next tile
    // go back to the first row using cr and then move by the number of contiguous faces in the tile (always 2 irrespective of unit_dim)
    addr_mod_pack_t {
        .y_src = {.incr = 2, .cr = 1},
    }
        .set(ADDR_MOD_1);

    // this address mod moves back to the beginning of the unit and separate instruction will increment the z counter to move to the next unit
    // unit here refers to the interleaved set of 4 * unit_dim faces (half in the top half of the active dest bank and half in the bottom half)
    addr_mod_pack_t {
        .y_src = {.clr = 1},
    }
        .set(ADDR_MOD_3);
}

inline void _llk_pack_fast_tilize_mop_config_(const std::uint32_t unit_dim)
{
    // UNPACR instructions are used with unit_dim 1 and 2 and SKIP instructions are used with unit_dim 3
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,
        false,
        TT_OP_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0),
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_NOP,
        TT_OP_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0),
        TT_OP_NOP,
        TT_OP_NOP);

    tmp.program(instrn_buffer);
}

template <DstSync Dst>
inline void _llk_pack_fast_tilize_init_(const std::uint32_t use_32bit_dest, const std::uint32_t pack_dst_format, const std::uint32_t unit_dim)
{
    // instead of using the actual is_fp32_dest_acc_en flag dest 32 bit mode is enabled if unpack_dst_format is TF32
    // this is due to a hw quirk with MOVA2D and MOVB2D
    // so clear PCK_DEST_RD_CTRL_Read_32b_data unless unpack_src_format is TF32
    // unpack src format is not easy to determine here so use an argument that is going to be computed at the higher level
    if (!use_32bit_dest)
    {
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
    }

    // set the address offset to the size of the tile in 16B words
    uint tile_size = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
    if (IS_BFP_FORMAT(pack_dst_format))
    {
        tile_size += (TILE_C_DIM * TILE_R_DIM) / 16; // one exp byte per 16 datums
    }
    tile_size = tile_size >> 4; // convert to 16B words
    TT_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, tile_size, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));

    // since faces are interleaved and the top and bottom faces are in the separate halves of the active dest bank, each packer needs a special offset
    // difference between 16 bit dest and 32 bit dest is where the half of the active bank is (256 rows vs 128 rows)
    // stallwait and select_packer_dest_registers just replicate what _llk_init_packer_dest_offset_registers_ does
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    if (!use_32bit_dest)
    {
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x100, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x101, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x100, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x101, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
    }
    else
    {
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x080, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 2));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, 0x000 + 0x081, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_LO + 3));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x000, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x001, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 1));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x080, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 2));
        TTI_SETDMAREG(p_setdmareg::PAYLOAD_IMMEDIATE, DEST_REGISTER_HALF_SIZE + 0x081, p_setdmareg::MODE_IMMEDIATE, LO_16(p_gpr_pack::DEST_OFFSET_HI + 3));
    }
    select_packer_dest_registers<Dst>();

    // each packer packs a single row per call and in total each packer will pack a single face
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    _llk_pack_fast_tilize_addrmod_config_(unit_dim);

    _llk_pack_fast_tilize_mop_config_(unit_dim);
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_tilize_uninit_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false)
{
    // restore PCK_DEST_RD_CTRL_Read_32b_data to the original value
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(is_fp32_dest_acc_en);

    // restore default packer dest offsets
    _llk_init_packer_dest_offset_registers_<Dst, DstTileFaceLayout::RowMajor>();

    // packers pack a whole face by default, restore it
    TTI_SETADCXX(p_setadc::PAC, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    // reset counters
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));

    // for some reason short inits avoid the packer init (probably since it is usually the same)
    // but that means calling it here with reasonable defaults is needed
    // it just initializes the address mods and mop
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile);
}

inline void _llk_pack_fast_tilize_block_(
    const std::uint32_t tile_index, const std::uint32_t address, const std::uint32_t unit_dim, const std::uint32_t num_units)
{
    // use false here so that the 31st bit of the address remains set as the offset addresses for the other packers continue to be used
    // while the address for the first packer is manipulated using ADDDMAREG and REG2FLOP
    program_packer_destination(address, false);

    // reset counters and set the W counter
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
    // move to the start tile index, instead of using the standard W counter whose stride is a single tile
    // use the Z counter whose stride is a single face as tiles are split into halves of the active dest bank
    // so only move 2 faces per tile_index
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index << 1);

    for (uint i = 0; i < num_units; i++)
    {
        if (unit_dim == 1)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in dest (same counter rationale as for tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 2); // CH0Z += 2
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // this pack should behave as a no op aside from the address mod side effect (which is resetting the Y counter to the beginning of the tile)
            // but it actually provides some kind of a stall required when modifying the L1 base address while the packer is running
            // and has less performance impact than a PACK PACK STALLWAIT
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
        else if (unit_dim == 2)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here moves to the next tile in the same unit
            TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0x0);
            TTI_PACR_COMMON(ADDR_MOD_0, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next unit in dest (2 * 2 faces, same thing as tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 4); // CH0Z += 4
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here resets to the beginning of the unit
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
        else if (unit_dim == 3)
        {
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here moves to the next tile in the same unit
            TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here moves to the next tile in the same unit
            TTI_PACR_COMMON(ADDR_MOD_1, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
            // pack a single tile
            // inside mop:
            // for (uint j = 0; j < 15; j++)
            // {
            //     TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 0);
            // }
            TTI_MOP(p_mop::MASK_LOOP, (FACE_R_DIM - 1) - 1, 0xFFFF);
            TTI_PACR_COMMON(ADDR_MOD_2, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 0, 1);
            // move to the next unit in dest (3 * 2 faces, same thing as tile_index)
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 6); // CH0Z += 6
            // move to the next tile in L1
            TTI_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
            TTI_REG2FLOP_COMMON(p_reg2flop::WRITE_4B, REG2FLOP_FLOP_INDEX(THCON_SEC0_REG1_L1_Dest_addr_ADDR32), p_gpr_pack::OUTPUT_ADDR);
            // same notes for the flush bit as above
            // address mod here resets to the beginning of the unit
            TTI_PACR_COMMON(ADDR_MOD_3, p_pacr::P_ZERO_OUTPUT_DISABLED, PACK_SEL(NUM_PACKERS), 1, 0);
        }
    }
}
