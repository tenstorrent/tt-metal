// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_pack_common_api.h"
#include "llk_pack_fast_tilize.h"

/*************************************************************************
 * LLK PACK FAST TILIZE
 *************************************************************************/

inline void llk_pack_fast_tilize_init(
    const std::uint32_t input_operand, const std::uint32_t pack_output, const std::uint32_t unit_dim) {
    const std::uint8_t input_id = get_output_id(input_operand);
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const std::uint32_t use_32bit_dest = pack_src_format[input_id] == (std::uint32_t)DataFormat::Float32 ||
                                         pack_src_format[input_id] == (std::uint32_t)DataFormat::Tf32;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    _llk_pack_fast_tilize_init_<DST_SYNC_MODE>(use_32bit_dest, pack_dst_format[output_id], unit_dim, num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_uninit(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(output_id);
    const bool partial_face = get_output_partial_face(output_id);

    _llk_pack_fast_tilize_uninit_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        pack_dst_format[output_id], tensor_shape, partial_face);
}

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
