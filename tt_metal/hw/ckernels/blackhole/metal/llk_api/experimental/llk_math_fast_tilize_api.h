// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_fast_tilize.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATH FAST TILIZE (BH)
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
    const std::uint32_t in_id = get_operand_id(icb);
    const std::uint32_t out_id = get_operand_id(ocb);
    return !(
        ckernel::fast_tilize_block_supported(
            get_operand_tile_r_dim(in_id),
            get_operand_tile_c_dim(in_id),
            get_operand_face_r_dim(in_id),
            get_operand_num_faces(in_id),
            block) &&
        ckernel::fast_tilize_block_supported(
            get_operand_tile_r_dim(out_id),
            get_operand_tile_c_dim(out_id),
            get_operand_face_r_dim(out_id),
            get_operand_num_faces(out_id),
            block));
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_init(const std::uint32_t operand) {
    SAN_HOOK(unsupported());
    // Default fast-tilize init configures DEST remap; use the skip-remap variant only when the caller hoists it.
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_init_skip_remap(const std::uint32_t operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_init_<is_fp32_dest_acc_en, false>(unpack_dst_format[operand_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_uninit(const std::uint32_t operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_block_(
    const std::uint32_t dst_index, const std::uint32_t operand, const std::uint32_t unit_dim) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(dst_index, unpack_dst_format[operand_id], unit_dim, num_faces);
}
