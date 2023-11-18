// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK TILIZE
*************************************************************************/

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {

    constexpr bool  within_face_16x16_transpose = false;
    constexpr StochRndMode stoch_rnd_mode = StochRndMode::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces
    );
}


template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure_disaggregated(
    const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand
    };
    llk_unpack_tilize_hw_configure<is_fp32_dest_acc_en>(&unpack_tilize_params);
}

inline void llk_unpack_tilize_mop_config(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    _llk_unpack_tilize_mop_config_(narrow_tile);
}

inline void llk_unpack_tilize_init(const std::uint32_t operand = 0, const std::uint32_t ct_dim = 0) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    // Save state of unpacker config for quick restore
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, THCON_SEC0_REG2_Out_data_format_ADDR32); // Save unpack config[0]
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32); // Save tile x dim per context

    _llk_unpack_tilize_init_(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        ct_dim,
        face_r_dim,
        narrow_tile
    );

}

inline void llk_unpack_tilize_uninit(const std::uint32_t face_r_dim = FACE_R_DIM) {
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim*FACE_C_DIM-1, 0x0);
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0); // Restore unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32,  p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1); // Restore tile x dim per context
}

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {

    std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor

    _llk_unpack_tilize_(
        base_address,
        tile_index,
        unpack_src_format[operand_id],
        block_ct_dim,
        face_r_dim,
        num_faces,
        narrow_tile
    );
}
