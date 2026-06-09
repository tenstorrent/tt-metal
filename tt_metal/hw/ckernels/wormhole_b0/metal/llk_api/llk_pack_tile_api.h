// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op.
 *
 * Derives the destination format and tile geometry from the output's circular buffer. The skip_*
 * template flags let a caller reuse state already established by a prior init or hw-configure.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @param pack_output: Circular-buffer index of the pack output.
 * @param num_tiles: Number of tiles processed per MOP run.
 * @ref llk_pack is the matching execute call.
 */
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t pack_output = 16, std::uint32_t num_tiles = 1) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Wormhole B0: pack init supports PackMode::Default and PackMode::Untilize only");
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    if constexpr (!skip_addrmod_config) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }

    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_dst_format[output_id], face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

/**
 * @brief Pack one tile from the destination register to the output circular buffer.
 *
 * Resolves the L1 destination address from the output's circular buffer (in order, or via
 * output_tile_index for out-of-order output), then runs the packer MOP.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam out_of_order_output: Address the output tile by output_tile_index rather than sequentially.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (out-of-order untilize is unsupported).
 * @param tile_index: Index of the source tile in the destination register.
 * @param output: Circular-buffer index of the pack output.
 * @param output_tile_index: Tile index within the output CB (used for out-of-order output).
 * @note Call @ref llk_pack_init with matching template/runtime args before this function.
 */
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr =
        get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

/**
 * @brief Pack a contiguous block of matmul output tiles from the destination register to L1.
 *
 * Repeats the single-tile pack over @p ntiles consecutive dest tiles, resolving the output address
 * from the output's circular buffer.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam out_of_order_output: Address the output tile by output_tile_index rather than sequentially.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (out-of-order untilize is unsupported).
 * @param start_tile_index: First source tile index in the destination register.
 * @param output: Circular-buffer index of the pack output.
 * @param ntiles: Number of consecutive tiles to pack.
 * @param output_tile_index: Tile index within the output CB (used for out-of-order output).
 * @note Call @ref llk_pack_init with matching template/runtime args before this function.
 */
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_matmul_pack(
    std::uint32_t start_tile_index, std::uint32_t output, uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    LLK_ASSERT(
        ((start_tile_index + ntiles - 1) < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr =
            get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
    }
}
