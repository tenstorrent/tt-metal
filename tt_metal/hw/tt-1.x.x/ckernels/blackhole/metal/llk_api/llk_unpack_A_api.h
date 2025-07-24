// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_A.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

template <
    bool is_fp32_dest_acc_en,
    StochRndType stoch_rnd_mode = StochRndType::None,
    bool disable_src_zero_flag = false>
inline void llk_unpack_A_hw_configure(
    const llk_unpack_A_params_t* unpack_A_params, const int within_face_16x16_transpose = 0) {
    const uint32_t unpA_operand_id = get_operand_id(unpack_A_params->unpA_operand);
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode, disable_src_zero_flag>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces);
}

template <
    bool is_fp32_dest_acc_en,
    StochRndType stoch_rnd_mode = StochRndType::None,
    bool disable_src_zero_flag = false>
inline void llk_unpack_A_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_A_params_t unpack_A_params = {.unpA_operand = unpA_operand};
    llk_unpack_A_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode, disable_src_zero_flag>(
        &unpack_A_params, within_face_16x16_transpose);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_mop_config(
    const bool transpose_of_faces,
    const std::uint32_t operand_id,
    const std::uint32_t unpack_src_format = 0,
    std::uint32_t unpack_dst_format = 0) {
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_unpack_A_mop_config_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    const std::uint32_t operand_unpack_src_format = unpack_src_format[operand_id];
    const std::uint32_t operand_unpack_dst_format = unpack_dst_format[operand_id];
    if (unpack_to_dest && is_32bit_input(operand_unpack_src_format, operand_unpack_dst_format)) {
        llk_unpack_dbg_feature_disable();
    }

    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces,
        within_face_16x16_transpose,
        face_r_dim,
        num_faces,
        operand_unpack_src_format,
        operand_unpack_dst_format);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(
    const std::uint32_t operand, const std::uint32_t tile_index, const bool transpose_of_faces = 0) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    WAYPOINT("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, transpose_of_faces > 0, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    WAYPOINT("UPAD");
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand,
    const std::uint32_t start_tile_index,
    const std::uint32_t ntiles,
    const bool transpose_of_faces = 0) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size;
    std::uint32_t address = base_address;

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
            address, transpose_of_faces > 0, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
        address += offset_address;
        WAYPOINT("UPAD");
    }
}
