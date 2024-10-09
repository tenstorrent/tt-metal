// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

template <bool untilize = false>
inline void _llk_pack_configure_addrmod_() {
    addr_mod_pack_t{
        .y_src = {.incr = untilize ? 0 : 1},
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
        .z_dst = {.incr = 1, .clr = 0},
    }
    .set(ADDR_MOD_1);


    if constexpr (untilize) {
       addr_mod_pack_t{
           .y_src = { .incr = 1, .clr = 0, .cr = 1  },
           .y_dst = { .incr = 1, .clr = 0, .cr = 0  },
       }.set(ADDR_MOD_2);
    }
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true>
inline void _llk_pack_mop_config_() {

    constexpr uint MOP_INNER_LOOP = 16;
    constexpr uint MOP_UNTILIZE_INNER_LOOP = FaceLayout == DstTileFaceLayout::ColMajor ? 8 : 4;
    constexpr uint MOP_OUTER_LOOP = 1;
    constexpr uint MOP_UNTILIZE_OUTER_LOOP = 8;
    constexpr uint PACKCNT = 4;
    constexpr uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    ckernel::ckernel_template tmp(
        untilize ? MOP_UNTILIZE_OUTER_LOOP : MOP_OUTER_LOOP, untilize ? MOP_UNTILIZE_INNER_LOOP : MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));


    if constexpr (!untilize) {
        tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 1));

        if constexpr (write_tile_header) {
            // Write header to l1
            tmp.set_end_op(TT_OP_STOREIND(
                1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));
        }
    } else {
        tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
        tmp.set_end_op(TT_OP_PACR(ADDR_MOD_2, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
        tmp.set_last_outer_loop_instr(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 4, 0));
    }

    tmp.program(instrn_buffer);
}

template <bool untilize = false>
inline void _llk_pack_hw_configure_(const uint pack_src_format, const uint pack_dst_format, const uint tile_size, const uint relu_config) {
    configure_pack<untilize>(pack_src_format, pack_dst_format, tile_size, relu_config);
}

// FIXME: Remove once edge mask spec is defined
template <bool untilize = false, PoolType type, ReduceDim dim>
inline void _llk_pack_reduce_hw_configure_(const uint pack_src_format, const uint pack_dst_format, const uint tile_size, const uint relu_config) {

    configure_pack<untilize>(pack_src_format, pack_dst_format, tile_size, relu_config);

    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();

    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        for (uint i = 0; i < 4; i++) cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i] = 0x00000001;
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = 0x00000000;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x00000001;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000001;
    } else {
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = 0x00000000;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = 0x0000ffff;

        if constexpr (untilize) {
            cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000005;
        } else {
            cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x00000001;
        }
    }
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor, bool write_tile_header = true>
inline void _llk_pack_init_() {
    _llk_pack_configure_addrmod_<untilize>();
    _llk_pack_mop_config_<untilize, zero_output, FaceLayout, write_tile_header>();
}

template <DstSync Dst, bool out_of_order_output = false, bool untilize = false, bool is_fp32_dest_acc_en = false /* unused*/>
inline void _llk_pack_(std::uint32_t tile_index, std::uint32_t pack_dst_format, std::uint16_t pack_tile_addr) {

    if constexpr (Dst == DstSync::SyncTile16) {
        // Z-counter points to the next tile in dest
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
        pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
    } else {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
    }

    program_packer_destination(pack_tile_addr, pack_dst_format);

    mop_run(1, 1);

    if constexpr (untilize) {
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
    }
}
