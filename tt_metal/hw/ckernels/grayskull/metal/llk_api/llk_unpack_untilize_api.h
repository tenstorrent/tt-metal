// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_untilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK UNTILIZE
*************************************************************************/
template <bool is_fp32_dest_acc_en = false /*not used*/>
inline void llk_unpack_untilize_hw_configure(const llk_unpack_A_params_t *unpack_untilize_params) {

    const uint32_t unpA_operand_id = get_operand_id(unpack_untilize_params->unpA_operand);

    _llk_unpack_untilize_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id]
    );
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure<is_fp32_dest_acc_en>(&unpack_untilize_params);
}

inline void llk_unpack_untilize_mop_config() {
    _llk_unpack_untilize_mop_config_();
}

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_unpack_untilize_init_(
        face_r_dim,
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        cb_interface[operand_id].fifo_page_size
    );
}

inline void llk_unpack_untilize_uninit(uint32_t operand) {
    DEBUG_STATUS("UPUW");
    std::uint32_t operand_id = get_operand_id(operand);
    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    // Get pointer to registers for current state ID
    volatile uint *cfg = get_cfg_pointer();

    TT_SETADCXX(p_setadc::UNP0, FACE_R_DIM*FACE_C_DIM-1, 0x0);

    unpack_tile_descriptor_u tile_descriptor;
    tile_descriptor.val[0] = 0;
    tile_descriptor.val[1] = 0;

    // Set descriptor 0
    tile_descriptor.f.in_data_format = (uint)unpack_src_format[operand_id];
    tile_descriptor.f.uncompressed = 1;
    tile_descriptor.f.x_dim = 256;

    // Set descriptor 1
    tile_descriptor.f.y_dim = 1;
    tile_descriptor.f.z_dim = 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_descriptor.val[1]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(tile_descriptor.val[1]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG0_TileDescriptor_ADDR32+1-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

    uint unpA_ch1_x_stride = (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float32   ? 4
                             : (uint)(unpack_dst_format[operand_id] & 0x3) == (uint)DataFormat::Float16 ? 2
                                                                                                          : 1;
    uint unpA_ch1_y_stride = 16*16*unpA_ch1_x_stride;
    uint reg_val = (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                   (            0 << UNP0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);
    TT_SETDMAREG(0, LOWER_HALFWORD(reg_val), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(reg_val), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);

    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, UNP0_ADDR_BASE_REG_0_Base_ADDR32); // Clear base address register
    TTI_NOP; TTI_NOP;
    DEBUG_STATUS("UPUD");
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(
        base_address,
        block_tile_cols
    );
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    DEBUG_STATUS("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    DEBUG_STATUS("UPUD");
}
