#pragma once
#include "llk_io_pack.h"
#include "llk_defs.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_pack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void _llk_pack_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false) {

    addr_mod_pack_t{
        .y_src = {.incr = untilize ? 0 : 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    if constexpr (untilize) {
       addr_mod_pack_t{
           .y_src = { .incr = 1, .clr = 0, .cr = 1  },
           .y_dst = { .incr = 1, .clr = 0, .cr = 0  },
       }.set(ADDR_MOD_1);
    } else {
       addr_mod_pack_t{
           .y_src = {.incr = 0, .clr = 1, .cr = 0},
           .y_dst = {.incr = 0, .clr = 1, .cr = 0},
           .z_src = {.incr = 0, .clr = 0},
           .z_dst = {.incr = 0, .clr = 0},
       }
       .set(ADDR_MOD_1);
    }

    addr_mod_pack_t{
        .y_src = { .incr = 0, .clr = 0, .cr = 0  },
        .y_dst = { .incr = 0, .clr = 0, .cr = 0  },
    }.set(ADDR_MOD_2);
    
    const uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    const uint PACKCNT = partial_face ? 1 : num_faces;
    
    // Write header to l1
    if constexpr (!untilize) {
        const uint MOP_INNER_LOOP = 1;
        const uint MOP_OUTER_LOOP = 1;
        
        ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));

        if (partial_face) {
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)); // Don't close the tile, point to the next face
            tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0)); // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            tmp.set_loop_op1(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1)); // Close the tile
        }
        // Write header to l1
        tmp.set_end_op(TT_OP_STOREIND(
            1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));

        tmp.program(instrn_buffer);
    } else {
        const uint MOP_UNTILIZE_INNER_LOOP = ((face_r_dim == 1) || narrow_tile) ? 1 : (FaceLayout == DstTileFaceLayout::ColMajor ? 8 : 4);
        const uint MOP_UNTILIZE_OUTER_LOOP = ((face_r_dim == 1) || narrow_tile) ? 1 : (face_r_dim >> 1);

        if ((face_r_dim == 1) || narrow_tile) {
            ckernel::ckernel_template tmp(MOP_UNTILIZE_OUTER_LOOP, MOP_UNTILIZE_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program(instrn_buffer);
        } else {
            ckernel::ckernel_template tmp(MOP_UNTILIZE_OUTER_LOOP, MOP_UNTILIZE_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)); 
            tmp.program(instrn_buffer);
        }
    }
}

template <bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_hw_configure_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false, const std::uint32_t relu_config = 0) {

    configure_pack<is_fp32_dest_acc_en, untilize>(
        pack_src_format,
        pack_dst_format,
        tile_size,
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile,
        relu_config
    );
}

template <bool untilize = false, PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_reduce_hw_configure_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t tile_size, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false, const std::uint32_t relu_config = 0) {

    configure_pack<is_fp32_dest_acc_en, untilize>(
        pack_src_format,
        pack_dst_format,
        tile_size,
        face_r_dim,
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

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void _llk_pack_init_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool partial_face = false, const bool narrow_tile = false) {

    _llk_pack_mop_config_<untilize, zero_output, FaceLayout>(
        face_r_dim, 
        num_faces, 
        partial_face, 
        narrow_tile
    );
}


template <DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address) {


    constexpr uint32_t DEST_NUM_TILES_SHIFT = is_fp32_dest_acc_en ? (1) : (0);
    constexpr uint32_t DEST_NUM_TILES = DEST_NUM_TILES_FP16 >> DEST_NUM_TILES_SHIFT;

    if constexpr (Dst == DstSync::SyncTile16) {
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

    if constexpr (untilize) {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close tile
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 1, 0);
    }
}