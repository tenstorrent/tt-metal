// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_untilize.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *************************************************************************/
template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_untilize_hw_configure(const llk_unpack_A_params_t *unpack_untilize_params) {
    constexpr bool is_row_pool = false;
    constexpr bool within_face_16x16_transpose = false;
    constexpr StochRndType stoch_rnd_mode = StochRndType::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_untilize_params->unpA_operand);
    const uint32_t unpA_num_faces = 4;
    const uint32_t unpA_face_r_dim = FACE_R_DIM;

    _llk_unpack_untilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure<is_fp32_dest_acc_en>(&unpack_untilize_params);
}

inline void llk_unpack_untilize_mop_config() { _llk_unpack_untilize_mop_config_(); }

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    // Save state of unpacker config for quick restore
    TTI_RDCFG(
        p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32);  // Save unpack stride config
    TTI_RDCFG(
        p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1,
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);  // Save tile x dim per context
    TTI_RDCFG(
        p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);  // Save descriptor 1

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], cb_interface[operand_id].fifo_page_size, face_r_dim, num_faces);
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
    TTI_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16);
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
    const std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}
