// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "llk_pack_untilize.h"

/*************************************************************************
 * LLK PACK UNTILIZE
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). They take
// already-resolved scalar formats/geometry + a runtime write address / per-row stride; the per-source
// prologue (resolving these from a CB id, or from an LLKMemDescriptor) lives in the callers. The CB-id
// callers pass base_addr = fifo_wr_ptr - 1 and page_stride = full_ct_dim * fifo_page_size; the id-free
// callers pass base_addr = cb_write_address(...) and a compile-time page_stride derived from the output
// descriptor (fifo_page_size == a single tile size assumption -- see experimental/2_0/llk_pack_untilize.h).

template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline void llk_pack_untilize_init_impl(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t num_faces) {
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format, pack_dst_format));
    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        pack_src_format, pack_dst_format, face_r_dim, num_faces);
}

template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
inline void llk_pack_untilize_impl(
    const std::uint32_t block_rt_dim,
    const std::uint32_t base_addr,
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t page_stride,
    const std::uint32_t face_r_dim,
    const std::uint32_t num_faces,
    const std::uint32_t block_c_index,
    const std::uint32_t tile_dst_rt_offset) {
    std::uint32_t pack_tile_addr =
        base_addr + SCALE_DATUM_SIZE(
                        pack_dst_format,
                        (block_c_index * ((num_faces > 2) ? num_faces / 2 : num_faces) * block_ct_dim * FACE_C_DIM)) /
                        16;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format, pack_dst_format));

    for (std::uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        _llk_pack_untilize_<block_ct_dim, full_ct_dim, narrow_row, tile_dst_ct_offset, dense>(
            pack_tile_addr, num_faces, block_rt * block_ct_dim + tile_dst_rt_offset);

        pack_tile_addr += page_stride;
    }
}

inline void llk_pack_untilize_uninit_impl(const std::uint32_t pack_src_format) {
    _llk_pack_untilize_uninit_(pack_src_format);
}

/**
 * Initialize the packer for an untilize operation on the given output operand.
 *
 * Face geometry (face_r_dim, num_faces) is derived from the output CB metadata. In
 * debug builds, validates that the packers are configured correctly for the resolved
 * face row dimension before programming the untilize init sequence.
 *
 * @tparam block_ct_dim   Width of a single block in tiles.
 * @tparam full_ct_dim    Width of the full input in tiles (defaults to block_ct_dim).
 * @tparam diagonal       Diagonal packing flag (unused on Blackhole; must be false).
 * @tparam narrow_row     Whether the input rows are narrow.
 * @tparam row_num_datums Number of datums per row.
 * @tparam dense          Pack two 2-face tiles into a single 4-face region.
 * @param  output         Output circular buffer / operand index.
 */
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false /* unused */,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline void llk_pack_untilize_init(std::uint32_t output) {
    static_assert(diagonal == false, "Diagonal is only supported on WH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    llk_pack_untilize_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        pack_src_format[output_id], pack_dst_format[output_id], face_r_dim, num_faces);
}

/**
 * Tear down the packer untilize configuration so a subsequent operation can
 * reprogram the packer. The source format is read from the output CB metadata.
 *
 * @param output Output circular buffer / operand index.
 */
inline void llk_pack_untilize_uninit(std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    llk_pack_untilize_uninit_impl(pack_src_format[output_id]);
}

/**
 * Pack an untilized block of tiles from the destination register into the output CB.
 *
 * Iterates over block_rt_dim tile rows, computing the packer write address from the
 * output CB fifo state for each row. Face geometry (face_r_dim, num_faces) is derived
 * from the output CB metadata.
 *
 * @tparam block_ct_dim       Width of a single block in tiles.
 * @tparam full_ct_dim        Width of the full input in tiles (defaults to block_ct_dim).
 * @tparam diagonal           Diagonal packing flag (unused on Blackhole; must be false).
 * @tparam narrow_row         Whether the input rows are narrow.
 * @tparam row_num_datums     Number of datums per row (unused on Blackhole).
 * @tparam tile_dst_ct_offset Compile-time column offset of the tile in the destination register.
 * @tparam dense              Pack two 2-face tiles into a single 4-face region.
 * @param  block_rt_dim       Height of the block in tiles (number of rows to pack).
 * @param  output             Output circular buffer / operand index.
 * @param  block_c_index      Block column index (used when full_ct_dim > block_ct_dim).
 * @param  tile_dst_rt_offset Runtime row offset of the tile in the destination register.
 */
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false /* unused */,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM /* unused */,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    std::uint32_t output,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    static_assert(diagonal == false, "Diagonal is only supported on WH");
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    llk_pack_untilize_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, tile_dst_ct_offset, dense>(
        block_rt_dim,
        get_local_cb_interface(output_id).fifo_wr_ptr - 1,
        pack_src_format[output_id],
        pack_dst_format[output_id],
        full_ct_dim * get_local_cb_interface(output_id).fifo_page_size,
        face_r_dim,
        num_faces,
        block_c_index,
        tile_dst_rt_offset);
}
