// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "llk_defs.h"

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_pack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool untilize = false, bool tilize = false>
inline void _llk_pack_configure_addrmod_() {

    if constexpr(untilize && !tilize) {

        /*  Y src & Y dest inc by 1 to give strided increments:
            Rows: 0, 16, 1, 17, 2, 18, ........ 15, 31
        */
        addr_mod_pack_t {
            .y_src = { .incr = 1 },
            .y_dst = { .incr = 1 },
            .z_src = { .incr = 0 },
            .z_dst = { .incr = 0 }
        }.set(ADDR_MOD_0);

        /* Increment Faces by 2 to give next 2 faces:
            Rows: 32, 48, 33, 49, 34, 50........47, 63
        */
        addr_mod_pack_t {
            .y_src = { .incr = 0, .clr = 1 },
            .y_dst = { .incr = 0, .clr = 1 },
            .z_src = { .incr = 1 },
            .z_dst = { .incr = 0 }
        }.set(ADDR_MOD_1);

        addr_mod_pack_t {
            .y_src = { .incr = 0, .clr = 1 },
            .y_dst = { .incr = 0, .clr = 1 },
            .z_src = { .incr = 0, .clr = 1 },
            .z_dst = { .incr = 0, .clr = 1 }
        }.set(ADDR_MOD_2);

    } else if constexpr(tilize && !untilize) {

        addr_mod_pack_t {
            .y_src = { .incr = 4 },
            .y_dst = { .incr = 2 },
            .z_src = { .incr = 0 },
            .z_dst = { .incr = 0 }
        }.set(ADDR_MOD_0);


        addr_mod_pack_t {
            .y_src = { .incr = 0, .clr = 1 },
            .y_dst = { .incr = 0, .clr = 1 },
            .z_src = { .incr = 0 },
            .z_dst = { .incr = 0 }
        }.set(ADDR_MOD_1);

        //Increment faces by 2 (jump 2 dest address 32)
        addr_mod_pack_t {
            .y_src = { .incr = 0, .clr = 1 },
            .y_dst = { .incr = 0, .clr = 1 },
            .z_src = { .incr = 1 },
            .z_dst = { .incr = 0 }
        }.set(ADDR_MOD_2);

    } else {

        addr_mod_pack_t{
            .y_src = {.incr = 4},
            .y_dst = {.incr = 4},
        }.set(ADDR_MOD_0);

        addr_mod_pack_t{
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 1},
            .z_dst = {.incr = 0, .clr = 0},
        }.set(ADDR_MOD_1);

        addr_mod_pack_t{
            .y_src = { .incr = 0, .clr = 1, .cr = 0  },
            .y_dst = { .incr = 4, .clr = 0, .cr = 0  },
            .z_src = { .incr = 1, .clr = 0  },
        }.set(ADDR_MOD_2);

    }

}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true, bool tilize = false>
inline void _llk_pack_mop_config_(const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false) {
    static_assert(FaceLayout == DstTileFaceLayout::RowMajor, "FaceLayout must be RowMajor");

    constexpr uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    if constexpr(untilize && !tilize) {

        const uint PACK_INTF_SEL = (tile_c_dim < TILE_C_DIM) ? p_pacr::SINGLE_INTF_ACTIVE : p_pacr::TWO_INTFS_ACTIVE;
        const uint MOP_INNER_LOOP = face_r_dim;
        const uint MOP_OUTER_LOOP = (tile_c_dim < TILE_C_DIM) ? num_faces : (num_faces >> 1);

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_STRIDED_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0)
        );

        tmp.set_last_inner_loop_instr(
            TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_STRIDED_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0)
        );
        tmp.set_last_outer_loop_instr(
            TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_STRIDED_MODE, ADDR_MOD_2, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, 0, p_pacr::NO_CTXT_CTRL, 0, 1)
        );
        tmp.program(instrn_buffer);

    } else if constexpr(tilize && !untilize) {

        const uint PACK_INTF_SEL_0 = 0b0101;
        const uint PACK_INTF_SEL_1 = 0b1010;
        const uint MOP_INNER_LOOP = 1;
        const uint MOP_OUTER_LOOP = 2;
        const uint replay_buf_len = 16;

        //This replay buffer finishes 2 faces
        load_replay_buf(0, replay_buf_len, false,
            // Lambda function to set up replay buffer
            [] {
                //Face 0 -> mask rows 1010
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);

                //Face 1 -> mask rows 0101
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0);
                TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_2, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, 0, p_pacr::NO_CTXT_CTRL, 0, 1);

            }
        );

        // ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0));
        // tmp.set_last_inner_loop_instr(TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_0, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0));
        // tmp.set_last_outer_loop_instr(TTI_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL_1, 0, MEGAROW, p_pacr::NO_CTXT_CTRL, 0, 0));

        ckernel::ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_REPLAY(0, replay_buf_len, 0, 0)
        );

        if constexpr (write_tile_header) {
            tmp.set_end_ops(
                TT_OP_SETADCZW(p_setadc::PAC, 0, 2, 0, 0, 0b0100), //ch0_z = 0, ch1_z = 2;
                TT_OP_STOREIND(1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR)); // write tile header to L1
        }
        else {
            tmp.set_end_op(
                TT_OP_SETADCZW(p_setadc::PAC, 0, 2, 0, 0, 0b0100)); //ch0_z = 0, ch1_z = 2;
        }

        tmp.program(instrn_buffer);

    } else {

        const uint PACK_INTF_SEL = face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

        const uint MOP_INNER_LOOP = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
        const uint MOP_OUTER_LOOP = num_faces;

        ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, 0, 0, 0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_2, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, 0, 0, 0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, PACK_INTF_SEL, 0, 0, 0, 0, 1));

        // if (partial_face) {
        //     tmp.set_start_op(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_0, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Don't close the tile, point to the next face
        //     tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0)); // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
        //     tmp.set_loop_op1(TT_OP_PACR(p_pacr::CFG_CTXT_0, p_pacr::NO_ROW_PAD_ZERO, p_pacr::DST_ACCESS_NORMAL_MODE, ADDR_MOD_1, p_pacr::ADDR_CNT_CTXT_0, ZERO_OUTPUT_FLAG, p_pacr::ALL_INTF_ACTIVE, 0, MEGAROW, 0, 0, 1)); // Close the tile
        // }
        // Write header to l1
        if constexpr (write_tile_header) {
            tmp.set_end_op(TT_OP_STOREIND(
                1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));
        }

        tmp.program(instrn_buffer);
    }


}

template <bool is_fp32_dest_acc_en = false, bool is_tile_dim_reconfig_en = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true>
inline void _llk_pack_reconfig_data_format_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false) {

    reconfig_packer_data_format<is_fp32_dest_acc_en>(
        pack_src_format,
        pack_dst_format,
        tile_size,
        face_r_dim,
        tile_c_dim
    );

    if constexpr (is_tile_dim_reconfig_en) {
        _llk_pack_mop_config_<false, false, FaceLayout, write_tile_header>(pack_dst_format, face_r_dim, tile_c_dim, num_faces, partial_face, narrow_tile);
    }
}

template <bool untilize = false, bool is_fp32_dest_acc_en = false, bool tilize = false>
inline void _llk_pack_hw_configure_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false, const std::uint32_t relu_config = 0) {

    configure_pack<is_fp32_dest_acc_en, untilize, tilize>(
        pack_src_format,
        pack_dst_format,
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        narrow_tile,
        relu_config
    );
}

template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_reduce_hw_configure_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false, const std::uint32_t relu_config = 0) {

    configure_pack<is_fp32_dest_acc_en, untilize, false>(
        pack_src_format,
        pack_dst_format,
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        narrow_tile,
        relu_config
    );

    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    ckernel::packer::pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask = 0x0;
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x0001;
        if constexpr (untilize) {
            pack_edge_offset.f.tile_row_set_select_pack0 = 1;
            pack_edge_offset.f.tile_row_set_select_pack1 = 1;
            pack_edge_offset.f.tile_row_set_select_pack2 = 1;
            pack_edge_offset.f.tile_row_set_select_pack3 = 1;
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x11111111; // each packer packs 1x32 row
        } else {
            pack_edge_offset.f.tile_row_set_select_pack0 = 1;
            pack_edge_offset.f.tile_row_set_select_pack2 = 1;
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x55555555; // each packer packs 1x16 row
        }
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = pack_edge_offset.val;
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = pack_edge_offset.val;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x0001;
        cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000001;
    } else {
        pack_edge_offset.f.tile_row_set_select_pack0 = 1;
        pack_edge_offset.f.tile_row_set_select_pack1 = 1;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = pack_edge_offset.val;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0xffff;

        if constexpr (untilize) {
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000005;// Each packer packs 1x32 row
        } else {
            cfg[TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32] = 0x00000001;
        }
    }
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true, bool tilize = false>
inline void _llk_pack_init_(const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t tile_c_dim = TILE_C_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false) {

    _llk_pack_configure_addrmod_<untilize, tilize>();

    _llk_pack_mop_config_<untilize, zero_output, FaceLayout, write_tile_header, tilize>(
        pack_dst_format,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        narrow_tile
    );
}

template <DstSync Dst, bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address) {

    if constexpr (Dst == DstSync::SyncTile16) {
        constexpr uint32_t DEST_NUM_TILES_SHIFT = is_fp32_dest_acc_en ? (1) : (0);
        constexpr uint32_t DEST_NUM_TILES = DEST_NUM_TILES_FP16 >> DEST_NUM_TILES_SHIFT;
        // W-counter points to the next tile in dest
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, pack_sync_tile_dst_ptr);
        pack_sync_tile_dst_ptr += 1;
        pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr & (DEST_NUM_TILES - 1);
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, pack_sync_tile_dst_ptr);
        pack_sync_tile_dst_ptr = 0;
    } else {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
    }

    program_packer_destination(address);

    mop_run(1, 1);

    TT_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101); //reset z counters
}

#include "llk_pack_untilize.h"
