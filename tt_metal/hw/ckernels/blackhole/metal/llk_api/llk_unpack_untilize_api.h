// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_untilize.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *************************************************************************/

inline void llk_unpack_untilize_mop_config() { _llk_unpack_untilize_mop_config_(); }

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

inline void llk_unpack_untilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t unpA_ch1_x_stride = (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float32   ? 4
                                      : (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float16 ? 2
                                                                                                                 : 1;
    std::uint32_t unpA_ch1_y_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;

    WAYPOINT("UPUW");
    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    // Wait for cfg to be free to edit
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    // Reset the values to default in unpack AB common.
    TT_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;  // Wait for WRCFG to complete (takes 2 cycles)
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xFFFF>(1);
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(unpA_ch1_y_stride);
    TTI_NOP;
    TTI_NOP;  // Do we need this for WH?
    WAYPOINT("UPUD");
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}
