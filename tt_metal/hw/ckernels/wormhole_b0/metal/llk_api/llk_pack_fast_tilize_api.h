// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_pack_fast_tilize.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK FAST TILIZE
 *************************************************************************/

/**
 * @brief Whether fast_tilize_init/block/uninit must take the plain tilize path for this block.
 *
 * The predicate does not check the L1 format of 32x32 tiles (input format support stays the caller contract:
 * the fast unpack takes Float16_b or Float32); 16x32 tiny tiles need an even width and Float16_b or Float32,
 * both with 16-row faces (see ckernel::fast_tilize_block_supported). Both the input and the
 * output CB are checked, tile shape, face geometry and L1 format, through this thread's arrays, which cover every CB
 * index and are generated from the same descriptor on every thread (the unpack-source and pack-destination L1 formats
 * of a CB are equalized by the JIT), so all three threads reach the same decision even when the two CBs differ.
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
            get_output_dst_format(in_id),
            block) &&
        ckernel::fast_tilize_block_supported(
            get_output_tile_r_dim(out_id),
            get_output_tile_c_dim(out_id),
            get_output_face_r_dim(out_id),
            get_output_num_faces(out_id),
            get_output_dst_format(out_id),
            block));
}

inline void llk_pack_fast_tilize_init(
    const std::uint32_t input_operand, const std::uint32_t pack_output, const std::uint32_t unit_dim) {
    const std::uint8_t input_id = get_output_id(input_operand);
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const uint32_t use_32bit_dest =
        pack_src_format[input_id] == (uint)DataFormat::Float32 || pack_src_format[input_id] == (uint)DataFormat::Tf32;

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    SAN_HOOK(init<OperationPackFastTilizeWh>(
        StateVal<OperationPackFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<OperationPackFastTilizeWh::Use32BitDest>(use_32bit_dest),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<std::uint32_t>(input_operand)));

    _llk_pack_fast_tilize_init_<DST_SYNC_MODE>(use_32bit_dest, pack_dst_format[output_id], unit_dim, num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_uninit(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    const bool partial_face = get_output_partial_face(output_id);
    const bool narrow_tile = get_output_narrow_tile(output_id);

    SAN_HOOK(uninit<OperationPackFastTilizeWh>(
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
        StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile)));

    _llk_pack_fast_tilize_uninit_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        pack_dst_format[output_id], face_r_dim, num_faces, partial_face, narrow_tile);

    // The uninit tail-calls _llk_pack_init_, so seat OperationPack on llk_pack_init's num_tiles == 1 path.
    if (partial_face) {
        SAN_HOOK(init<OperationPack>(
            StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
            StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
            StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
            StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
            StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile),
            StateVal<OperationPack::NumTiles>(1)));
    } else {
        SAN_HOOK(init<OperationPack>(
            StateDiscard<std::uint32_t>(pack_dst_format[output_id]),
            StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
            StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
            StateVal<Operand<Exu::Pack>::PartialFace>(partial_face),
            StateVal<Operand<Exu::Pack>::NarrowTile>(narrow_tile),
            StateVal<OperationPack::NumTiles>(1)));
    }
}

inline void llk_pack_fast_tilize_block(
    const std::uint32_t tile_index,
    const std::uint32_t output,
    const std::uint32_t output_tile_index,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units) {
    LLK_ASSERT(
        (tile_index < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");

    const std::uint8_t output_id = get_output_id(output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const std::uint32_t pack_tile_addr = get_output_tile_address<true, PackMode::Default>(output_id, output_tile_index);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    SAN_HOOK(execute<OperationPackFastTilizeWh>(
        StateVal<OperationPackFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<std::uint32_t>(tile_index),
        StateDiscard<std::uint32_t>(output_tile_index),
        StateDiscard<std::uint32_t>(num_units)));

    _llk_pack_fast_tilize_block_(tile_index, pack_tile_addr, unit_dim, num_units, num_faces);
}
