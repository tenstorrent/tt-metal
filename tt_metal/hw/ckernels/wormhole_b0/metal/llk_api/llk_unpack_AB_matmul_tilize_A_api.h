// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_matmul_tilize_A.h"
#include "llk_unpack_common_api.h"

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_matmul_tilize_A_hw_configure(const llk_unpack_AB_matmul_params_t* unpack_AB_params) {
    // StochRndType stoch_rnd_mode = StochRndType::None
    // const std::uint32_t transpose_xy_srca = 0
    const uint32_t unpA_operand_id = get_operand_id(unpack_AB_params->unpB_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_AB_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpB_num_faces = get_operand_num_faces(unpB_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(unpB_operand_id);
    _llk_unpack_AB_matmul_tilize_A_hw_configure<is_fp32_dest_acc_en>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_AB_matmul_tilize_A_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    // StochRndType stoch_rnd_mode = StochRndType::None
    // const std::uint32_t transpose_xy_srca = 0
    const llk_unpack_AB_matmul_params_t unpack_AB_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_AB_matmul_tilize_A_hw_configure<is_fp32_dest_acc_en>(&unpack_AB_params);
}

inline void llk_unpack_AB_matmul_tilize_A_mop_config() {
    // const bool transpose = 0
    // const bool partial_face_a = 0
    // const bool partial_face_b = 0
    _llk_unpack_AB_matmul_tilize_A_mop_config();
}

__attribute__((always_inline)) inline void llk_unpack_AB_matmul_tilize_A_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t reuse_a = 0) {
    // const std::uint32_t transpose = 0
    const uint32_t operandA_id = get_operand_id(operandB);
    const uint32_t operandB_id = get_operand_id(operandA);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);
    const uint32_t unpB_num_faces = get_operand_num_faces(operandB_id);
    _llk_unpack_AB_matmul_tilize_A_init(
        ct_dim, rt_dim, kt_dim, unpA_face_r_dim, unpB_face_r_dim, unpA_num_faces, unpB_num_faces, reuse_a);
}

inline void llk_unpack_AB_matmul_tilize_A(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t reuse_a = 0) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandB_id);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    _llk_unpack_AB_matmul_tilize_A(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        unpA_face_r_dim,
        unpB_face_r_dim,
        ct_dim,
        rt_dim,
        kt_dim,
        reuse_a);
}
