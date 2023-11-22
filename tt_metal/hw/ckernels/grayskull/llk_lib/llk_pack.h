// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
inline void llk_pack_mop_config() {


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

    const uint MOP_INNER_LOOP = 16;
    const uint MOP_UNTILIZE_INNER_LOOP = FaceLayout == DstTileFaceLayout::ColMajor ? 8 : 4;
    const uint MOP_OUTER_LOOP = 1;
    const uint MOP_UNTILIZE_OUTER_LOOP = 8;
    const uint PACKCNT = 4;
    const uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;

    ckernel::ckernel_template tmp(
        untilize ? MOP_UNTILIZE_OUTER_LOOP : MOP_OUTER_LOOP, untilize ? MOP_UNTILIZE_INNER_LOOP : MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));


    if constexpr (!untilize) {
        tmp.set_last_inner_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, 0, 0, 1));

        // there's no tile headers, so we don't need to do this
        // Write header to l1
        //tmp.set_end_op(TT_OP_STOREIND(
        //    1, 0, p_ind::LD_16B, LO_16(0), p_ind::INC_NONE, p_gpr_pack::TILE_HEADER, p_gpr_pack::OUTPUT_ADDR));
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
inline void llk_pack_hw_configure(const llk_pack_params_t *pack_params) {
    configure_pack<untilize>(get_output_id(pack_params->pack_output), pack_params->relu_config.val);
}

inline void llk_pack_reconfig_data_format(const std::uint32_t old_operand, const std::uint32_t new_operand) {
    std::uint32_t old_operand_id = get_output_id(old_operand);
    std::uint32_t new_operand_id = get_output_id(new_operand);

    if((pack_dst_format[old_operand_id] != pack_dst_format[new_operand_id])
       && (pack_dst_format[old_operand_id] != (uint)DataFormat::Invalid)
       && (pack_dst_format[new_operand_id] != (uint)DataFormat::Invalid)) {
        reconfig_packer_data_format(new_operand_id);
    }
}

template <bool untilize = false, ReluType relu_type=ReluType::NO_RELU, std::uint32_t relu_threshold=0>
inline void llk_pack_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_hw_configure<untilize>(&llk_pack_params);
}

// FIXME: Remove once edge mask spec is defined
template <bool untilize = false, PoolType type, ReduceDim dim>
inline void llk_pack_reduce_hw_configure(const llk_pack_params_t *pack_params) {
    configure_pack<untilize>(get_output_id(pack_params->pack_output), pack_params->relu_config.val);
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

template <bool untilize = false, PoolType type, ReduceDim dim, ReluType relu_type=ReluType::NO_RELU, std::uint32_t relu_threshold=0>
inline void llk_pack_reduce_hw_configure_disaggregated(std::uint32_t pack_output) {
    llk_pack_params_t llk_pack_params = {
        .pack_output = pack_output, .relu_config = {.f = {.ApplyRelu = (std::uint32_t)relu_type, .Threshold = relu_threshold}}};
    llk_pack_reduce_hw_configure<untilize, type, dim>(&llk_pack_params);
}

template <bool untilize = false, bool zero_output = false, DstTileFaceLayout FaceLayout = DstTileFaceLayout::RowMajor>
inline void llk_pack_init() {
    llk_pack_mop_config<untilize, zero_output, FaceLayout>();
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false>
inline void llk_matmul_pack(std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);
    const std::uint8_t OUTPUT_BASE_ID = (std::uint8_t) get_output_base_id();

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    for (uint32_t tile_index=start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {

        std::uint16_t pack_tile_addr;
        if constexpr (out_of_order_output) {
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
                            MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID], (std::uint16_t)output_tile_index);
        } else {
            // in-order pack: 1) start with wr_ptr and then increment fifo_wr_tile_ptr tile by tile
            // note: packer is programmed to automatically skip the tile header
            // however, since there is no tile header we need to -1 the pack address (in terms of 16B words) to offset packer's +1
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr - 1;
            cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[OUTPUT_BASE_ID]);
        }

        if constexpr (Dst == DstSync::SyncTile16) {
            // Z-counter points to the next tile in dest
        } else if constexpr (Dst == DstSync::SyncTile2) {
            TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
            pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
        } else {
            TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
        }

        program_packer_destination(pack_tile_addr, OUTPUT_BASE_ID);

        mop_run(1, 1);

        if constexpr (untilize) {
            TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
            TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
        }
    }
}

template <bool out_of_order_output = false, bool untilize = false>
inline std::uint16_t get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {

    std::uint16_t pack_tile_addr;
    if constexpr (out_of_order_output) {
        pack_tile_addr = cb_interface[output_id].fifo_wr_ptr +
                         MUL_TILE_SIZE_AND_INDEX((std::uint8_t)pack_dst_format[output_id], (std::uint16_t)output_tile_index);
    } else {
        if constexpr (untilize) {
            // TODO: uplift this option from BBE
        } else {
            pack_tile_addr = cb_interface[output_id].fifo_wr_ptr + cb_interface[output_id].fifo_wr_tile_ptr;
            cb_interface[output_id].fifo_wr_tile_ptr += GET_L1_TILE_SIZE((std::uint8_t)pack_dst_format[output_id]);
        }
    }
    return pack_tile_addr - 1;
}

template <bool out_of_order_output = false, DstSync Dst = SyncFull, bool untilize = false, bool is_fp32_dest_acc_en = false /* unused*/>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    // Todo: figure out tile dims based on output
    std::uint8_t output_id = get_output_id(output);

    static_assert((!(untilize && out_of_order_output)) && "untilize out of order packing is not supported!");

    std::uint16_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(output_id, output_tile_index);

    if constexpr (Dst == DstSync::SyncTile16) {
        // Z-counter points to the next tile in dest
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, pack_sync_tile_dst_ptr);
        pack_sync_tile_dst_ptr = pack_sync_tile_dst_ptr + 8;
    } else {
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, tile_index);
    }

    program_packer_destination(pack_tile_addr, output_id);

    mop_run(1, 1);

    if constexpr (untilize) {
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 0);
        TTI_INCADCZW(p_setadc::PAC, 0, 0, 0, 1);
    }
}


template <ReduceDim dim, bool at_kernel_start = false, bool revert=false>
inline void llk_pack_reduce_config_v2(uint32_t icb_out) {

    if constexpr (at_kernel_start)
        configure_pack(get_output_id(icb_out), false);
    else {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
        tensix_sync();
    }

    volatile uint *cfg = get_cfg_pointer();
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        for (uint i = 0; i < 4; i++)
            //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+i);
            cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i] = revert ? 0xFFFFffff : 0x1;
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x1;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
    } else {
        //TTI_WRCFG(revert ? 0xFFFFffff : 0x0,    p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0);
        //TTI_WRCFG(revert ? 0xFFFFffff : 0xFFFF, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1);
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+0] = revert ? 0xFFFFffff : 0x0;
        cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32+1] = revert ? 0xFFFFffff : 0x0000ffff;
        cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = revert ? 0xF : 0x1;
    }
}

#include "llk_pack_untilize.h"
