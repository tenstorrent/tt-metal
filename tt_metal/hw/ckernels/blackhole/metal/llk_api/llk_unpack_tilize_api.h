// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_A_params_t *unpack_tilize_params) {
    constexpr bool within_face_16x16_transpose = false;
    constexpr StochRndType stoch_rnd_mode = StochRndType::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_tilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_tilize_params = {.unpA_operand = unpA_operand};
    llk_unpack_tilize_hw_configure<is_fp32_dest_acc_en>(&unpack_tilize_params);
}

inline void llk_unpack_tilize_mop_config(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    _llk_unpack_tilize_mop_config_(narrow_tile);
}

inline void llk_unpack_tilize_init(const std::uint32_t operand, const std::uint32_t ct_dim) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    _llk_unpack_tilize_init_(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], ct_dim, face_r_dim, narrow_tile);
}

inline void llk_unpack_tilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    // Revert X dim value to default.
    TT_SETADCXX(p_setadc::UNP_A, face_r_dim * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, face_r_dim * FACE_C_DIM - 1, 0x0);

    // Revert Z dim value back to default.
    const uint Tile_z_dim = get_operand_num_faces(operand);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 16, 0xffff0000>(Tile_z_dim);

    std::uint32_t operand_id = get_operand_id(operand);
    unpack_config_u config = {0};

    config.f.out_data_format = (uint)unpack_dst_format[operand_id];
    config.f.throttle_mode = 2;
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    // Load unpack config[0]
    TTI_WRCFG(p_gpr_unpack::TMP0,0,THCON_SEC0_REG2_Out_data_format_ADDR32);
    // GPR preloaded with  16 | (16 << 16)}
    TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_NOP;
}

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    std::uint32_t base_address =
        get_local_cb_interface(operand_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor

    WAYPOINT("UPTW");
    _llk_unpack_tilize_(
        base_address, tile_index, unpack_src_format[operand_id], block_ct_dim, face_r_dim, num_faces, narrow_tile);
    WAYPOINT("UPTD");
}

inline void llk_unpack_tilize_block(std::uint32_t operand, std::uint32_t block_c_tiles) {
    for (std::uint32_t tile_index = 0; tile_index < block_c_tiles; tile_index++) {
        llk_unpack_tilize(operand, tile_index, block_c_tiles);
    }
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_tilizeA_B_hw_configure(
    const llk_unpack_AB_params_t *unpack_tilizeA_B_params, const int within_face_16x16_transpose = 0) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpack_tilizeA_B_params->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_tilizeA_B_params->unpB_operand);

    // unpA -> srcA
    // Unpack only 1x16 row of datums to SrcA per UNPACK instruction
    const uint32_t num_faces_a = get_operand_num_faces(unpA_operand_id);
    const uint32_t face_r_dim_a = get_operand_face_r_dim(unpA_operand_id);

    // unpB -> srcB
    const uint32_t num_faces_b = get_operand_num_faces(unpB_operand_id);
    const uint32_t face_r_dim_b = get_operand_face_r_dim(unpB_operand_id);
    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        face_r_dim_a,
        face_r_dim_b,
        within_face_16x16_transpose,
        num_faces_a,
        num_faces_b);
}

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_tilizeA_B_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_AB_params_t unpack_tilizeA_B_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_tilizeA_B_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(
        &unpack_tilizeA_B_params, within_face_16x16_transpose);
}

// TODO: add support for all the template parameters
template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_mop_config(const bool narrow_tile = false, const std::uint32_t num_faces = 4) {
    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(narrow_tile, num_faces);
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces = 4,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM) {

    const std::uint32_t operand_id = get_operand_id(operandA);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    _llk_unpack_tilizeA_B_init_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        narrow_tile,
        ct_dim,
        num_faces,
        unpA_face_r_dim,
        unpB_face_r_dim
    );
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4,
    std::uint32_t unpA_face_r_dim = FACE_R_DIM) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);

    // TODO: RT face_r_dim should be taken from get_operand_face_r_dim(operandA_id);
    // But currently ops do not populate that array correctly
    const std::uint32_t face_r_dim = unpA_face_r_dim;

    const std::uint32_t base_address_a =
        get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    const std::uint32_t base_address_b =
        get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor
    const std::uint32_t offset_address_b = tile_index_b * get_local_cb_interface(operandB_id).fifo_page_size;
    const std::uint32_t address_b = base_address_b + offset_address_b;

    WAYPOINT("UPTW");

    _llk_unpack_tilizeA_B_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format[operandA_id],
        face_r_dim,
        narrow_tile,
        base_address_a,
        address_b,
        tile_index_a,
        tile_index_b,
        block_ct_dim,
        num_faces
    );

    WAYPOINT("UPTD");
}

template <bool neginf_srcA = false, std::uint32_t reload_srcB = false, bool zero_srcA = false, bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_block(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t block_c_tiles_a,
    std::uint32_t tile_idx_b,
    std::uint32_t num_faces = 4,
    std::uint32_t unpA_face_r_dim = FACE_R_DIM) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces, unpA_face_r_dim);
    }
}
