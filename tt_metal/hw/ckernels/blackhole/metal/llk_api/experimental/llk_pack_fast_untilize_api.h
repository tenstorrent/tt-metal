// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_pack_common_api.h"
#include "experimental/llk_pack_fast_untilize.h"

/*************************************************************************
 * LLK PACK FAST UNTILIZE (BH)
 *************************************************************************/

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_init_with_formats(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format) {
    ckernel::_llk_pack_fast_untilize_init_<block_ct_dim, full_ct_dim>(pack_src_format, pack_dst_format);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_init(const std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    LLK_ASSERT(
        get_output_num_faces(output_id) == ckernel::FAST_UNTILIZE_NUM_FACES,
        "fast_untilize pack requires 4-face output");
    ckernel::_llk_pack_fast_untilize_init_<block_ct_dim, full_ct_dim>(
        pack_src_format[output_id], pack_dst_format[output_id]);
}

template <std::uint32_t block_ct_dim>
inline void llk_pack_fast_untilize_block_at_address(
    const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim) {
    ckernel::_llk_pack_fast_untilize_block_<block_ct_dim>(address, unit_dim, prev_unit_dim);
}

template <std::uint32_t block_ct_dim>
inline void llk_pack_fast_untilize_block(
    const std::uint32_t output, const std::uint32_t output_tile_index, const std::uint32_t unit_dim) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t output_address = get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);
    std::uint32_t prev_unit_dim = 0;
    ckernel::_llk_pack_fast_untilize_block_<block_ct_dim>(output_address, unit_dim, prev_unit_dim);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_block_strided_at_address(
    const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim) {
    ckernel::_llk_pack_fast_untilize_block_strided_<block_ct_dim, full_ct_dim>(address, unit_dim, prev_unit_dim);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_block_strided(
    const std::uint32_t output,
    const std::uint32_t output_tile_index,
    const std::uint32_t tile_offset,
    const std::uint32_t unit_dim,
    std::uint32_t& prev_unit_dim) {
    const std::uint32_t output_id = get_output_id(output);
    const std::uint32_t output_row_address =
        get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);
    const std::uint32_t chunk_offset = SCALE_DATUM_SIZE(pack_dst_format[output_id], tile_offset * TILE_C_DIM) / 16;
    const std::uint32_t output_row_stride = SCALE_DATUM_SIZE(pack_dst_format[output_id], full_ct_dim * TILE_C_DIM) / 16;
    ckernel::_llk_pack_fast_untilize_block_strided_<block_ct_dim, full_ct_dim>(
        output_row_address + chunk_offset, unit_dim, prev_unit_dim, output_row_stride);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_uninit_with_src_format(const std::uint32_t pack_src_format) {
    ckernel::_llk_pack_fast_untilize_uninit_<block_ct_dim, full_ct_dim>(pack_src_format);
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void llk_pack_fast_untilize_uninit(const std::uint32_t output) {
    const std::uint32_t output_id = get_output_id(output);
    ckernel::_llk_pack_fast_untilize_uninit_<block_ct_dim, full_ct_dim>(pack_src_format[output_id]);
}
