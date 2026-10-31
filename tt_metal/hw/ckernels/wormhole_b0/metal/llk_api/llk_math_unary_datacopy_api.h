// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_fast_tilize.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(std::uint32_t dst_index, std::uint32_t operand) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand);
    SAN_HOOK(execute<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(get_operand_num_faces(operand_id)),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id]),
        StateDiscard<std::uint32_t>(dst_index)));

    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id], get_operand_tensor_shape(operand_id));
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    std::uint32_t start_dst_index, std::uint32_t ntiles, std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    SAN_HOOK(execute<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(get_operand_num_faces(operand_id)),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id]),
        StateDiscard<std::uint32_t>(start_dst_index),
        StateDiscard<std::uint32_t>(ntiles)));

    for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

        _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
            dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id], tensor_shape);
    }
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize || pack_mode == PackMode::Tilize,
        "Wormhole B0 math datacopy init: use PackMode::Default, PackMode::Untilize, or PackMode::Tilize (tilize is "
        "ignored on WH)");
    (void)pack_mode;
    const std::uint32_t operand_id = get_operand_id(operand);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    SAN_HOOK(init<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(get_operand_num_faces(operand_id)),
        StateVal<Operand<Exu::Fpu>::Format>(dst_format)));

    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(
        tensor_shape, dst_format);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    SAN_HOOK(uninit<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateDiscard<bool>(unpack_to_dest)));
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}

/*************************************************************************
 * LLK FAST ELTWISE UNARY DATACOPY
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
    const std::uint32_t in_id = get_operand_id(icb);
    const std::uint32_t out_id = get_operand_id(ocb);
    return !(
        ckernel::fast_tilize_block_supported(
            get_operand_tile_r_dim(in_id),
            get_operand_tile_c_dim(in_id),
            get_operand_face_r_dim(in_id),
            get_operand_num_faces(in_id),
            get_operand_src_format(in_id),
            block) &&
        ckernel::fast_tilize_block_supported(
            get_operand_tile_r_dim(out_id),
            get_operand_tile_c_dim(out_id),
            get_operand_face_r_dim(out_id),
            get_operand_num_faces(out_id),
            get_operand_src_format(out_id),
            block));
}

inline void llk_math_fast_tilize_init(const std::uint32_t operand, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    SAN_HOOK(init<OperationFpuFastTilizeWh>(
        StateVal<OperationFpuFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id])));

    _llk_math_fast_tilize_init_(unpack_dst_format[operand_id], unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    SAN_HOOK(uninit<OperationFpuFastTilizeWh>(StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id])));

    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

inline void llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    const std::uint32_t operand,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    SAN_HOOK(execute<OperationFpuFastTilizeWh>(
        StateVal<OperationFpuFastTilizeWh::AddrMod>(unit_dim == 1 ? 1 : 2),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id]),
        StateDiscard<std::uint32_t>(dst_index),
        StateDiscard<std::uint32_t>(num_units)));

    _llk_math_fast_tilize_block_(dst_index, unpack_dst_format[operand_id], unit_dim, num_units, num_faces);
}
