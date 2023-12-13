// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK TILIZE
*************************************************************************/

template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);

    _llk_unpack_tilize_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id]
    );
}

template <bool is_fp32_dest_acc_en = false /* unused */>
inline void llk_unpack_tilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand
    };
    llk_unpack_tilize_hw_configure(&unpack_tilize_params);
}

inline void llk_unpack_tilize_mop_config() {
    _llk_unpack_tilize_mop_config_();
}

inline void llk_unpack_tilize_init(const std::uint32_t operand=0, const std::uint32_t ct_dim=0) {


    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t src_format = (std::uint32_t)unpack_src_format[operand_id];
    std::uint32_t dst_format = (std::uint32_t)unpack_dst_format[operand_id];

    _llk_unpack_tilize_init_(src_format, dst_format, ct_dim);
}

//TODO: verify this function
inline void llk_unpack_tilize_uninit(const std::uint32_t operand=0/* not used*/, const std::uint32_t face_r_dim = FACE_R_DIM /* not used*/) {
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0); // Restore unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32,  p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1); // Restore tile x dim per context
}

// inline void llk_unpack_tilize_init(const std::uint32_t operand=0, const std::uint32_t ct_dim=0) {

//     wait_for_idle();
//     const std::uint32_t block_c_dim = ct_dim * TILE_C_DIM;

//     // Override default settings
//     std::uint32_t input = get_operand_id(operand);
//     unpack_config_u config = {0};

//     config.f.out_data_format = (uint)unpack_dst_format[input];
//     config.f.throttle_mode = 2;
//     config.f.tileize_mode = 1;
//     config.f.shift_amount = (SCALE_DATUM_SIZE((uint)unpack_src_format[input], block_c_dim)) >> 4;

//     TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
//     TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
//     TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
//     TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

//     llk_unpack_tilize_mop_config();
// }

// inline void llk_unpack_tilize_uninit(const std::uint32_t operand=0, const std::uint32_t face_r_dim = FACE_R_DIM /* not used*/) {
//     std::uint32_t input = get_operand_id(operand);
//     unpack_config_u config = {0};

//     config.f.out_data_format = (uint)unpack_dst_format[input];
//     config.f.throttle_mode = 2;
//     TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
//     TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
//     TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
//     TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_16x16); //GPR preloaded with  16 | (16 << 16)}
// }


inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t src_format = (uint)unpack_src_format[operand_id];

    _llk_unpack_tilize_(
        base_address,
        tile_index,
        src_format,
        block_ct_dim
    );
}

inline void llk_unpack_tilize_block(std::uint32_t operand, std::uint32_t block_c_tiles) {
    for (std::uint32_t tile_index = 0; tile_index < block_c_tiles; tile_index++) {
        llk_unpack_tilize(operand, tile_index, block_c_tiles);
    }
}
