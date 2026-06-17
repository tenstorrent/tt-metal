// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 * Initialize the unpacker for a single-operand tilize operation.
 *
 * Face row dimension, narrow-tile flag and face count are derived from the operand's CB metadata.
 *
 * @param operand Input circular buffer / operand index.
 * @param ct_dim  Number of tiles along the column (tilize block width).
 */
inline void llk_unpack_tilize_init(const std::uint32_t operand, const std::uint32_t ct_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_unpack_tilize_init_(
        unpack_src_format[operand_id], unpack_dst_format[operand_id], ct_dim, face_r_dim, narrow_tile, num_faces);
}

/**
 * Tear down the tilize unpacker configuration so a subsequent operation can reprogram the unpacker.
 *
 * @param operand Input circular buffer / operand index.
 */
inline void llk_unpack_tilize_uninit(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_tilize_uninit_((uint)unpack_dst_format[operand_id]);
}

/**
 * Unpack and tilize a single tile from the operand's circular buffer into srcA.
 *
 * Face geometry and narrow-tile flag are derived from the operand's CB metadata; the source base
 * address is read from the CB fifo state.
 *
 * @param operand      Input circular buffer / operand index.
 * @param tile_index   Tile index within the input to tilize.
 * @param block_ct_dim Number of tiles along the column of the tilize block.
 */
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

/**
 * Unpack and tilize a contiguous block of tiles by repeatedly calling llk_unpack_tilize.
 *
 * @param operand          Input circular buffer / operand index.
 * @param block_c_tiles    Number of column tiles in the block.
 * @param input_tile_index Starting tile index within the input (defaults to 0).
 */
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

/**
 * Program the unpacker MOP (macro-op) configuration for the combined tilize-A / unpack-B operation.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  num_faces        Number of faces per tile (defaults to 4).
 */
template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_mop_config(const std::uint32_t num_faces = 4) {
    _llk_unpack_tilizeA_B_mop_config_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(num_faces);
}

/**
 * Initialize the unpacker for the combined tilize-A / unpack-B operation.
 *
 * Operand A and B face geometry (face_r_dim, num_faces) and narrow-tile flag are derived from
 * circular-buffer unpack metadata (see set_unpack_face_geometry). In debug builds, validates that
 * both unpackers are configured consistently before programming the init sequence.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA         Input operand index for tilize source A.
 * @param  operandB         Input operand index for unpack source B.
 * @param  ct_dim           Number of tiles along the column (tilize block width).
 */
template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        num_faces,
        get_operand_num_faces(operandB_id)));

    _llk_unpack_tilizeA_B_init_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        narrow_tile,
        ct_dim,
        num_faces,
        unpA_face_r_dim,
        unpB_face_r_dim);
}

/**
 * @deprecated Operand A face geometry (num_faces, face_r_dim) is now derived from CB unpack metadata. Use the
 * llk_unpack_tilizeA_B_init(operandA, operandB, ct_dim) overload instead. This explicit-face-geometry overload
 * is retained only for backwards compatibility and will be removed.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA         Input operand index for tilize source A.
 * @param  operandB         Input operand index for unpack source B.
 * @param  ct_dim           Number of tiles along the column (tilize block width).
 * @param  num_faces        Number of faces per tile.
 * @param  unpA_face_r_dim  Face row dimension for operand A.
 * @param  unpB_face_r_dim  Face row dimension for operand B.
 */
template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
[[deprecated(
    "Operand A face geometry is now derived from CB unpack metadata; use the "
    "llk_unpack_tilizeA_B_init(operandA, operandB, ct_dim) overload instead.")]] inline void
llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim,
    const std::uint32_t num_faces,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[get_operand_id(operandB)],
        unpack_dst_format[get_operand_id(operandB)],
        unpA_face_r_dim,
        unpB_face_r_dim,
        num_faces,
        get_operand_num_faces(get_operand_id(operandB))));

    _llk_unpack_tilizeA_B_init_<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        narrow_tile,
        ct_dim,
        num_faces,
        unpA_face_r_dim,
        unpB_face_r_dim);
}

/**
 * Unpack and tilize one srcA tile while unpacking the corresponding srcB tile.
 *
 * Operand A face geometry and narrow-tile flag are derived from CB unpack metadata; source base
 * addresses are read from the CB fifo state.
 *
 * @tparam zero_srcA    Zero out srcA.
 * @param  operandA     Input operand index for tilize source A.
 * @param  operandB     Input operand index for unpack source B.
 * @param  tile_index_a Tile index within operand A.
 * @param  tile_index_b Tile index within operand B.
 * @param  block_ct_dim Number of column tiles in the block.
 */
template <bool zero_srcA = false>
inline void llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    std::uint32_t base_address_a =
        get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor

    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_b =
        get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t offset_address_b = tile_index_b * get_local_cb_interface(operandB_id).fifo_page_size;
    std::uint32_t address_b = base_address_b + offset_address_b;

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        face_r_dim,
        get_operand_face_r_dim(operandB_id),
        num_faces,
        get_operand_num_faces(operandB_id)));

    WAYPOINT("UPTW");
    _llk_unpack_tilizeA_B_<zero_srcA>(
        unpack_src_format[operandA_id],
        face_r_dim,
        narrow_tile,
        base_address_a,
        address_b,
        tile_index_a,
        block_ct_dim,
        num_faces);
    WAYPOINT("UPTD");
}

/**
 * @deprecated Operand A num_faces is now derived from CB unpack metadata. Use the
 * llk_unpack_tilizeA_B(operandA, operandB, tile_index_a, tile_index_b, block_ct_dim) overload instead. This
 * explicit-num_faces overload is retained only for backwards compatibility and will be removed.
 *
 * @tparam zero_srcA    Zero out srcA.
 * @param  operandA     Input operand index for tilize source A.
 * @param  operandB     Input operand index for unpack source B.
 * @param  tile_index_a Tile index within operand A.
 * @param  tile_index_b Tile index within operand B.
 * @param  block_ct_dim Number of column tiles in the block.
 * @param  num_faces    Number of faces per tile.
 */
template <bool zero_srcA = false>
[[deprecated(
    "Operand A num_faces is now derived from CB unpack metadata; use the "
    "llk_unpack_tilizeA_B(operandA, operandB, tile_index_a, tile_index_b, block_ct_dim) overload "
    "instead.")]] inline void
llk_unpack_tilizeA_B(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t tile_index_a,
    std::uint32_t tile_index_b,
    std::uint32_t block_ct_dim,
    std::uint32_t num_faces) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);
    const bool narrow_tile = get_operand_narrow_tile(operandA_id);

    std::uint32_t base_address_a =
        get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor

    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_b =
        get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t offset_address_b = tile_index_b * get_local_cb_interface(operandB_id).fifo_page_size;
    std::uint32_t address_b = base_address_b + offset_address_b;

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly<UnpackerProgramType::ProgramByFace>(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        face_r_dim,
        get_operand_face_r_dim(operandB_id),
        num_faces,
        get_operand_num_faces(operandB_id)));

    WAYPOINT("UPTW");
    _llk_unpack_tilizeA_B_<zero_srcA>(
        unpack_src_format[operandA_id],
        face_r_dim,
        narrow_tile,
        base_address_a,
        address_b,
        tile_index_a,
        block_ct_dim,
        num_faces);
    WAYPOINT("UPTD");
}

/**
 * Unpack and tilize a block of srcA column tiles against srcB by repeatedly calling
 * llk_unpack_tilizeA_B.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA        Input operand index for tilize source A.
 * @param  operandB        Input operand index for unpack source B.
 * @param  block_c_tiles_a Number of column tiles in operand A's block.
 * @param  tile_idx_b      Tile index within operand B.
 */
template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
inline void llk_unpack_tilizeA_B_block(
    std::uint32_t operandA, std::uint32_t operandB, std::uint32_t block_c_tiles_a, std::uint32_t tile_idx_b) {
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<zero_srcA>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a);
    }
}

/**
 * @deprecated Operand A face geometry (num_faces, face_r_dim) is now derived from CB unpack metadata. Use the
 * llk_unpack_tilizeA_B_block(operandA, operandB, block_c_tiles_a, tile_idx_b) overload instead. This
 * explicit-face-geometry overload is retained only for backwards compatibility and will be removed.
 *
 * @tparam neginf_srcA      Initialize srcA padding with negative infinity (for reduce-max).
 * @tparam reload_srcB      Whether srcB is reloaded each iteration.
 * @tparam zero_srcA        Zero out srcA.
 * @tparam zero_srcA_reduce Zero out srcA for the reduce path.
 * @param  operandA        Input operand index for tilize source A.
 * @param  operandB        Input operand index for unpack source B.
 * @param  block_c_tiles_a Number of column tiles in operand A's block.
 * @param  tile_idx_b      Tile index within operand B.
 * @param  num_faces       Number of faces per tile.
 * @param  unpA_face_r_dim Face row dimension for operand A (unused on Wormhole).
 */
template <
    bool neginf_srcA = false,
    std::uint32_t reload_srcB = false,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
[[deprecated(
    "Operand A face geometry is now derived from CB unpack metadata; use the "
    "llk_unpack_tilizeA_B_block(operandA, operandB, block_c_tiles_a, tile_idx_b) overload "
    "instead.")]] inline void
llk_unpack_tilizeA_B_block(
    std::uint32_t operandA,
    std::uint32_t operandB,
    std::uint32_t block_c_tiles_a,
    std::uint32_t tile_idx_b,
    std::uint32_t num_faces,
    [[maybe_unused]] std::uint32_t unpA_face_r_dim) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<zero_srcA>(operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a, num_faces);
    }
#pragma GCC diagnostic pop
}

/*************************************************************************
 * LLK UNPACK FAST TILIZE SRC A
 *************************************************************************/

/**
 * Initialize the unpacker for the fast (hardware-accelerated) srcA tilize path.
 *
 * @param operand  Input circular buffer / operand index.
 * @param full_dim Full row dimension of the input being tilized.
 */
inline void llk_unpack_fast_tilize_init(const std::uint32_t operand, std::uint32_t full_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);

    _llk_unpack_fast_tilize_init_(unpack_dst_format[operand_id], full_dim);
}

/**
 * Tear down the fast srcA tilize configuration.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_unpack_fast_tilize_uninit() {
    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

/**
 * Unpack and tilize a block of tiles using the fast srcA tilize path.
 *
 * Face count is derived from the operand's CB metadata; the source base address is read from the
 * CB fifo state.
 *
 * @param operand    Input circular buffer / operand index.
 * @param tile_index Starting tile index within the input.
 * @param unit_dim   Column dimension of a single tilize unit.
 * @param num_units  Number of tilize units in the block.
 * @param full_dim   Full row dimension of the input being tilized.
 */
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

/**
 * Tear down the combined tilize-A / unpack-B configuration so a subsequent operation can reprogram
 * the unpacker.
 *
 * @param operand Input circular buffer / operand index.
 */
inline void llk_unpack_tilizeA_B_uninit(const std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_tilizeA_B_uninit_((uint)unpack_dst_format[operand_id]);
}
