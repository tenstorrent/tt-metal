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

inline std::uint32_t llk_pack_fast_tilize_use_32bit_dest(const std::uint32_t input_operand) {
    const std::uint8_t input_id = get_output_id(input_operand);
    return pack_src_format[input_id] == (uint)DataFormat::Float32 ||
           pack_src_format[input_id] == (uint)DataFormat::Tf32;
}

// Packer input format programmed by _llk_pack_fast_tilize_init_: with fp32_dest_acc_en, a 16-bit dest read and a
// Float32/Tf32 output CB pack_src_format it switches the packer input to Float16_b, otherwise it keeps pack_src_format.
template <bool is_fp32_dest_acc_en>
inline std::uint32_t llk_pack_fast_tilize_pack_src_format(
    const std::uint32_t input_operand, const std::uint8_t output_id) {
    if constexpr (is_fp32_dest_acc_en) {
        if (_llk_pack_fast_tilize_switches_pack_src_(
                llk_pack_fast_tilize_use_32bit_dest(input_operand), pack_src_format[output_id])) {
            return (uint)DataFormat::Float16_b;
        }
    }
    return pack_src_format[output_id];
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_init(
    const std::uint32_t input_operand, const std::uint32_t pack_output, const std::uint32_t unit_dim) {
    const std::uint8_t output_id = get_output_id(pack_output);
    const std::uint32_t num_faces = get_output_num_faces(output_id);

    const uint32_t use_32bit_dest = llk_pack_fast_tilize_use_32bit_dest(input_operand);

    LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));

    SAN_HOOK(init<OperationPackFastTilizeWh>(
        StateVal<OperationPackFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<OperationPackFastTilizeWh::Use32BitDest>(use_32bit_dest),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<std::uint32_t>(input_operand)));

    _llk_pack_fast_tilize_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>(
        use_32bit_dest,
        pack_dst_format[output_id],
        unit_dim,
        num_faces,
        TILE_C_DIM * TILE_R_DIM,
        pack_src_format[output_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_uninit(const std::uint32_t input_operand, const std::uint32_t pack_output) {
    const std::uint32_t use_32bit_dest = llk_pack_fast_tilize_use_32bit_dest(input_operand);
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
        pack_dst_format[output_id],
        face_r_dim,
        num_faces,
        partial_face,
        narrow_tile,
        pack_src_format[output_id],
        use_32bit_dest);

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

template <bool is_fp32_dest_acc_en>
inline void llk_pack_fast_tilize_block(
    const std::uint32_t input_operand,
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

    // the fast tilize init may have switched the packer input format (see llk_pack_fast_tilize_pack_src_format)
    LLK_ASSERT_BLOCK(are_packers_configured_correctly(
        llk_pack_fast_tilize_pack_src_format<is_fp32_dest_acc_en>(input_operand, output_id),
        pack_dst_format[output_id]));

    SAN_HOOK(execute<OperationPackFastTilizeWh>(
        StateVal<OperationPackFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<Operand<Exu::Pack>::OutputFormat>(pack_dst_format[output_id]),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<std::uint32_t>(tile_index),
        StateDiscard<std::uint32_t>(output_tile_index),
        StateDiscard<std::uint32_t>(num_units),
        StateDiscard<std::uint32_t>(input_operand)));

    _llk_pack_fast_tilize_block_(tile_index, pack_tile_addr, unit_dim, num_units, num_faces);
}
