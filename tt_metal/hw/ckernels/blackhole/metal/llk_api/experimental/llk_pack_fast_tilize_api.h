// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_pack_common_api.h"
#include "experimental/llk_pack_fast_tilize.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK FAST TILIZE (BH)
 *************************************************************************/

/**
 * @brief Whether fast_tilize_init/block/uninit must take the plain tilize path for this block.
 *
 * Blackhole fast-tilize is a 32x32-tile path with no unit_dim 1 (see ckernel::fast_tilize_block_supported). Both the
 * input and the output CB are checked, tile shape and face geometry, through this thread's geometry arrays, which
 * cover every CB index and are generated from the same descriptor on every thread, so all three threads reach the
 * same decision even when the two CBs differ.
 *
 * @param icb: Input circular buffer identifier.
 * @param ocb: Output circular buffer identifier.
 * @param block: Block width in tiles (full_dim at init/uninit).
 */
inline bool llk_fast_tilize_needs_fallback(
    const std::uint32_t icb, const std::uint32_t ocb, const std::uint32_t block) {
    const std::uint32_t in_id = get_output_id(icb);
    const std::uint32_t out_id = get_output_id(ocb);
    return !(
        ckernel::fast_tilize_block_supported(
            get_output_tile_r_dim(in_id),
            get_output_tile_c_dim(in_id),
            get_output_face_r_dim(in_id),
            get_output_num_faces(in_id),
            block) &&
        ckernel::fast_tilize_block_supported(
            get_output_tile_r_dim(out_id),
            get_output_tile_c_dim(out_id),
            get_output_face_r_dim(out_id),
            get_output_num_faces(out_id),
            block));
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_init(
    const std::uint32_t input_operand, const std::uint32_t pack_output, const std::uint32_t unit_dim) {
    SAN_HOOK(unsupported());
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const uint32_t use_32bit_dest =
        pack_src_format[output_id] == (uint)DataFormat::Float32 || pack_src_format[output_id] == (uint)DataFormat::Tf32;
    _llk_pack_fast_tilize_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        use_32bit_dest, pack_dst_format[output_id], unit_dim, num_faces, pack_src_format[output_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_uninit(const std::uint32_t pack_output) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    _llk_pack_fast_tilize_uninit_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        pack_dst_format[output_id], face_r_dim, num_faces, pack_src_format[output_id]);
}

inline void llk_pack_fast_tilize_reinit_unit_dim(const std::uint32_t pack_output, const std::uint32_t new_unit_dim) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(pack_output);
    _llk_pack_fast_tilize_reinit_unit_dim_(pack_dst_format[output_id], new_unit_dim);
}

inline void llk_pack_fast_tilize_block(
    const std::uint32_t tile_index,
    const std::uint32_t output,
    const std::uint32_t output_tile_index,
    const std::uint32_t unit_dim) {
    SAN_HOOK(unsupported());
    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const std::uint32_t pack_tile_addr = get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);
    _llk_pack_fast_tilize_block_(tile_index, pack_tile_addr, unit_dim, num_faces);
}

// Row-scoped pack helpers: program destination once per row (row_begin),
// stream chunks without reprogramming (row_chunk), cleanup (row_end).
inline void llk_pack_fast_tilize_row_begin(const std::uint32_t output, const std::uint32_t output_tile_index) {
    SAN_HOOK(unsupported());
    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t pack_tile_addr = get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);
    _llk_pack_fast_tilize_row_begin_(pack_tile_addr);
}

inline void llk_pack_fast_tilize_row_chunk(
    const std::uint32_t tile_index, const std::uint32_t output, const std::uint32_t unit_dim) {
    SAN_HOOK(unsupported());
    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    _llk_pack_fast_tilize_row_chunk_(tile_index, unit_dim, num_faces);
}

inline void llk_pack_fast_tilize_row_end() {
    SAN_HOOK(unsupported());
    _llk_pack_fast_tilize_row_end_();
}
