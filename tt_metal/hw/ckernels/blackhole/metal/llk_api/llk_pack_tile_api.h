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
 * Derives the source format and face geometry from the output's circular buffer. When packing
 * tilized 8-bit datums the Blackhole row-unswizzle workaround is skipped (the issue does not
 * affect 8-bit datums).
 *
 * @tparam pack_mode: Packing layout, values = <Default/Tilize/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @param pack_output: Output circular-buffer index to configure the packer for.
 * @param num_tiles: Number of tiles processed per MOP run.
 * @param input_operand: Input operand circular-buffer index, used to detect 8-bit source datums.
 * @note Call @ref llk_pack with matching template args after this.
 */
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(
    const std::uint32_t pack_output = 16, std::uint32_t num_tiles = 1, const std::uint32_t input_operand = 0) {
    // TODO (https://github.com/tenstorrent/tt-metal/issues/18948): Revisit for narrow_tile
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    if constexpr (!skip_addrmod_config) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }

    // For pack with tilize enabled, check if the original input format is 8-bit.
    // 8-bit datums (Int8, UInt8, Fp8_e4m3, Lf8) do not require the tilize workaround on Blackhole.
    const std::uint32_t src_format = static_cast<std::uint32_t>(unpack_src_format[input_operand]);
    const bool is_input_8bit_format = IS_8BIT_FORMAT(src_format);
    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_format[output_id], face_r_dim, tile_c_dim, num_faces, num_tiles, is_input_8bit_format);
}

/**
 * @brief Pack one tile from the destination register to L1.
 *
 * Resolves the output tile's L1 write address from its circular buffer and runs the packer MOP.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam out_of_order_output: When true, address by output_tile_index; otherwise use the running fifo pointer.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (Tilize not supported here)
 * @param tile_index: Index of the source tile in the destination register.
 * @param output: Output circular-buffer index to write to.
 * @param output_tile_index: Tile index within the output (used only for out-of-order output).
 * @note Call @ref llk_pack_init with matching template/runtime args before this function.
 */
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr =
        get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

/**
 * @brief Pack a contiguous block of tiles from the destination register to L1 (matmul output path).
 *
 * Loops the packer MOP over @p ntiles destination tiles, resolving each output tile's L1 write
 * address from its circular buffer.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam out_of_order_output: When true, address by output_tile_index; otherwise use the running fifo pointer.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (Tilize not supported here)
 * @param start_tile_index: First source tile index in the destination register.
 * @param output: Output circular-buffer index to write to.
 * @param ntiles: Number of consecutive tiles to pack.
 * @param output_tile_index: Tile index within the output (used only for out-of-order output).
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
