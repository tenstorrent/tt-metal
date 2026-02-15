// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

inline void llk_unpack_tilize_mop_config(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    _llk_unpack_tilize_mop_config_(narrow_tile);
}

inline void llk_unpack_tilize_init(const std::uint32_t operand, const std::uint32_t ct_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);

    _llk_unpack_tilize_init_(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], ct_dim, face_r_dim, narrow_tile);
}

inline void llk_unpack_tilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_tilize_uninit_((uint)unpack_dst_format[operand_id], face_r_dim);
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
        base_address,
        tile_index,
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        block_ct_dim,
        face_r_dim,
        num_faces,
        narrow_tile);
    WAYPOINT("UPTD");
}

inline void llk_unpack_tilize_block(
    std::uint32_t operand, std::uint32_t block_c_tiles, std::uint32_t input_tile_index = 0) {
    // Not sure if input_tile_index can be arbitrary but it works for moving across rows of files,
    // i.e. input_tile_index % block_c_tiles == 0
    input_tile_index =
        input_tile_index % block_c_tiles + (input_tile_index / block_c_tiles) * block_c_tiles * TILE_R_DIM;
    for (std::uint32_t tile_index = 0; tile_index < block_c_tiles; tile_index++) {
        llk_unpack_tilize(operand, input_tile_index + tile_index, block_c_tiles);
    }
}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_mop_config(const bool narrow_tile = false, const std::uint32_t num_faces = 4) {
    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(narrow_tile, num_faces);
}

template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces = 4,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM) {
    const std::uint32_t operand_id =
        get_operand_id(operandA);  // Use operandA to get operand_id tile dims must be the same for both operands
    // const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
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

template <bool zero_srcA = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces = 4)
{
    std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    // const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    std::uint32_t base_address_a =
        get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor

    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_b =
        get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t offset_address_b = tile_index_b * get_local_cb_interface(operandB_id).fifo_page_size;
    std::uint32_t address_b = base_address_b + offset_address_b;

    WAYPOINT("UPTW");
    _llk_unpack_tilizeA_B_<zero_srcA>(
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
    std::uint32_t unpA_face_r_dim = FACE_R_DIM /*unused*/) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<zero_srcA>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces);
    }
}

/*************************************************************************
 * LLK UNPACK FAST TILIZE SRC A
 *************************************************************************/

inline void llk_unpack_fast_tilize_init(const std::uint32_t operand, std::uint32_t full_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);

    _llk_unpack_fast_tilize_init_(unpack_dst_format[operand_id], full_dim);
}

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_tilize_uninit() {
    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

inline void llk_unpack_fast_tilize_block(
    const std::uint32_t operand,
    const std::uint32_t tile_index,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units,
    const std::uint32_t full_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_fast_tilize_block_(
        base_address, tile_index, unpack_src_format[operand_id], unit_dim, num_units, full_dim, num_faces);
}

inline void llk_unpack_tilizeA_B_uninit(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_unpack_tilizeA_B_uninit_((uint)unpack_dst_format[operand_id], face_r_dim);
}
