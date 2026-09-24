// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/). They take
// already-resolved scalar formats/geometry + the runtime write address; the per-source prologue
// (resolving these from a CB id, or from an MemDescriptor) lives in the callers.
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init_impl(
    const std::uint32_t pack_src_reg_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool is_input_8bit_format) {
    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_reg_format, face_r_dim, tile_c_dim, num_faces, num_tiles, is_input_8bit_format);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_impl(std::uint32_t tile_index, std::uint32_t pack_tile_addr) {
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(
    const std::uint32_t pack_output, const std::uint32_t num_tiles, const std::uint32_t input_operand) {
    // TODO (https://github.com/tenstorrent/tt-metal/issues/18948): Revisit for narrow_tile
    const std::uint32_t output_id = get_output_id(pack_output);

    if constexpr (!skip_addrmod_config) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }

    // For pack with tilize enabled, check if the original input format is 8-bit.
    // 8-bit datums (Int8, UInt8, Fp8_e4m3, Lf8) do not require the tilize workaround on Blackhole.
    bool is_input_8bit_format = false;
    if constexpr (pack_mode == PackMode::Tilize) {
        is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(unpack_src_format[input_operand]));
    }
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        num_tiles,
        is_input_8bit_format);
}

// input_operand is only consumed by the Blackhole tilize workaround. Keep the common non-tilize API
// architecture-agnostic without giving the input CB id a default value.
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles = 1) {
    static_assert(pack_mode != PackMode::Tilize, "PackMode::Tilize requires an explicit input_operand");
    llk_pack_init<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_output, num_tiles, 0 /* input_operand unused outside PackMode::Tilize */);
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr =
        get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    llk::san::pack_operand_check(
        is_fp32_dest_acc_en,
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        llk::san::IGNORE,
        llk::san::IGNORE);

    llk_pack_impl<is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
}

template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_matmul_pack(
    std::uint32_t start_tile_index, std::uint32_t output, std::uint32_t ntiles, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);

    static_assert(
        !((pack_mode == PackMode::Untilize) && out_of_order_output), "untilize out of order packing is not supported!");
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    LLK_ASSERT(
        ((start_tile_index + ntiles - 1) < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    llk::san::pack_operand_check(
        is_fp32_dest_acc_en,
        pack_src_format[output_id],
        pack_dst_format[output_id],
        get_output_face_r_dim(output_id),
        get_output_tile_c_dim(output_id),
        get_output_num_faces(output_id),
        llk::san::IGNORE,
        llk::san::IGNORE);

    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        std::uint32_t pack_tile_addr =
            get_output_tile_address<out_of_order_output, pack_mode>(output_id, output_tile_index);

        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode>(tile_index, pack_tile_addr);
    }
}
