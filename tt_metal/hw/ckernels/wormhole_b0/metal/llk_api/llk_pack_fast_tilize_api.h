// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack_fast_tilize.h"

/*************************************************************************
 * LLK PACK FAST TILIZE
 *************************************************************************/

/**
 * @brief Initialize the packer for a fast-tilize pack op.
 *
 * Derives the destination format and face count from the output's circular buffer, and selects 32-bit
 * dest read mode when the input source format is FP32/TF32. Only DstSync::SyncHalf is supported;
 * supported output formats are FP32, FP16_B, BFP8_B, and BFP4_B.
 *
 * @param input_operand: Circular-buffer index of the input, whose source format selects 32-bit dest mode.
 * @param pack_output: Circular-buffer index of the pack output.
 * @param unit_dim: Number of tiles processed per iteration, valid values = <1, 2, 3>
 * @note On the unpack thread, pair with @ref llk_unpack_fast_tilize_init and on the math thread with
 *       @ref llk_math_fast_tilize_init (same unit_dim).
 * @note Pair with @ref llk_pack_fast_tilize_uninit after the matching @ref llk_pack_fast_tilize_block execute calls.
 */
inline void llk_pack_fast_tilize_init(
    const std::uint32_t input_operand, const std::uint32_t pack_output, const std::uint32_t unit_dim) {
    const std::uint8_t input_id = get_output_id(input_operand);
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const uint32_t use_32bit_dest =
        pack_src_format[input_id] == (uint)DataFormat::Float32 || pack_src_format[input_id] == (uint)DataFormat::Tf32;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_fast_tilize_init_<DST_SYNC_MODE>(use_32bit_dest, pack_dst_format[output_id], unit_dim, num_faces);
}

/**
 * @brief Tear down the packer after a fast-tilize pack op and restore default pack state.
 *
 * Derives the tile geometry from the output's circular buffer, restores the dest read mode and default
 * packer state, and re-runs the standard packer init so a subsequent (non fast-tilize) pack starts clean.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @param pack_output: Circular-buffer index of the pack output.
 * @note Call @ref llk_pack_fast_tilize_init before this function.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_uninit(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    _llk_pack_fast_tilize_uninit_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        pack_dst_format[output_id], face_r_dim, num_faces, partial_face, narrow_tile);
}

/**
 * @brief Fast-tilize-pack a block of units from the destination register to L1.
 *
 * Resolves the L1 destination address and face count from the output's circular buffer, then packs
 * num_units units (each covering unit_dim tiles), advancing the dest and L1 destination per unit.
 *
 * @param tile_index: Index of the first source tile in the destination register.
 * @param output: Circular-buffer index of the pack output.
 * @param output_tile_index: Tile index within the output CB.
 * @param unit_dim: Number of tiles processed per unit, valid values = <1, 2, 3>
 * @param num_units: Number of units to pack in this call.
 * @note Call @ref llk_pack_fast_tilize_init before this function, and @ref llk_pack_fast_tilize_uninit
 *       once all fast-tilize-pack calls are complete.
 * @note On the math thread, @ref llk_math_fast_tilize_block_ must have written the split top/bottom-half faces into
 * dest.
 */
inline void llk_pack_fast_tilize_block(
    const std::uint32_t tile_index,
    const std::uint32_t output,
    const std::uint32_t output_tile_index,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units) {
    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const std::uint32_t pack_tile_addr = get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_fast_tilize_block_(tile_index, pack_tile_addr, unit_dim, num_units, num_faces);
}
