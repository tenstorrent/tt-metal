// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_unary_broadcast.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * @brief Initialize eltwise unary datacopy operations
 *
 * For an unpack-to-dest operand the unpacker writes DEST directly and math has no MOP to
 * program; that case is gated on the `unpack_to_dest` template parameter and the math init is skipped.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE uses unary-broadcast math init when type is B2D
 * @tparam unpack_to_dest Set true to enable the UNP_DEST routing decision (unary, non-broadcast path)
 * @tparam acc_to_dest Accumulate the broadcast into dest instead of overwriting it; broadcast path only, and must match
 * llk_unpack_A_init
 * @param operand Logical dataflow buffer id for the input operand
 * @note With EN_32BIT_DEST, an Int16 operand needs unpack_to_dest, including the B2D broadcast path.
 *       The FPU cannot datacopy Int16 into a 32-bit Dest. The check below is LLK_ASSERT-only.
 */
template <
    DataCopyType type,
    bool EN_32BIT_DEST,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false,
    bool is_int_fpu_en /*maybe_unused*/ = false,
    bool tilize /*maybe_unused*/ = false,
    bool acc_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    // Int16 is MOV-only, so this 32-bit Dest datacopy writes bad data. Unpack-to-Dest skips it.
    // LLK_ASSERT only: a normal build does not check. ProgramSpec rejects this input.
    if constexpr (EN_32BIT_DEST && !unpack_to_dest) {
        LLK_ASSERT(
            srcA_format != DataFormat::Int16,
            "Quasar FPU cannot datacopy Int16 into a 32-bit Dest; unpack this operand to Dest");
    }
    _configure_alu_cfg_<AluCfgType::DEFAULT, EN_32BIT_DEST>(srcA_format, srcB_format);

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        static_assert(!acc_to_dest, "acc_to_dest is supported only for unary broadcast");
        // Unpack-to-dest: the unpacker writes Dest under a loan, so math has nothing to do.
        if constexpr (!unpack_to_dest) {
            _llk_math_eltwise_unary_datacopy_init_<type, EN_32BIT_DEST>(
                num_rows /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        }
    } else {
        static_assert(type == DataCopyType::B2D);
        static_assert(!unpack_to_dest, "unpack_to_dest is not supported for unary broadcast");

        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand);
        _llk_math_eltwise_unary_broadcast_init_<src_b_bcast_type, EN_32BIT_DEST, false /*unpack_to_dest*/, acc_to_dest>(
            tensor_shape, srcB_format);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * For the non-broadcast path, when `unpack_to_dest` is true the unpacker writes DEST directly and
 * this call does nothing. The kernel must wrap the copy in tile_regs_borrow and tile_regs_return.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE with unpack_to_dest false uses unary-broadcast math
 * @tparam unpack_to_dest when true, unpack-to-dest path; plain datacopy otherwise
 * @param dst_index Tile index into the destination register.
 * @param operand Logical dataflow buffer id for the input operand.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool EN_32BIT_DEST = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    if constexpr (src_b_bcast_type != BroadcastType::NONE) {
        static_assert(type == DataCopyType::B2D, "Unary broadcast math path requires DataCopyType::B2D");
        static_assert(!unpack_to_dest, "unpack_to_dest is not supported for unary broadcast");
        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand);
        LLK_ASSERT(
            (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
            "llk_math_eltwise_unary_datacopy: dst_index exceeds dest capacity");
        _llk_math_eltwise_unary_broadcast_(dst_index);
    } else {
        // Unpack-to-dest: the unpacker writes Dest under a loan, so math has nothing to do.
        if constexpr (!unpack_to_dest) {
            LLK_ASSERT(
                (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
                "llk_math_eltwise_unary_datacopy: dst_index exceeds dest capacity");
            _llk_math_eltwise_unary_datacopy_(dst_index);
        }
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a block of tiles.
 *
 * When `unpack_to_dest` is true the unpacker writes DEST directly and this call does nothing. The
 * kernel must wrap the copy in tile_regs_borrow and tile_regs_return.
 *
 * @param start_dst_index Starting tile index in the destination register.
 * @param ntiles Number of tiles to copy to the destination register.
 * @param operand Logical dataflow buffer id for the input operand.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool EN_32BIT_DEST = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    // Unpack-to-dest: the unpacker writes Dest under a loan, so math has nothing to do.
    if constexpr (!unpack_to_dest) {
        LLK_ASSERT(
            (start_dst_index + ntiles <= get_dest_max_tiles<DST_SYNC_MODE, EN_32BIT_DEST>()),
            "llk_math_eltwise_unary_datacopy_block: dst range exceeds dest capacity");
        for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
            _llk_math_eltwise_unary_datacopy_(dst_index);
        }
    }
}

template <
    BroadcastType src_b_bcast_type /*maybe_unused*/ = BroadcastType::NONE,
    bool unpack_to_dest /*maybe_unused*/ = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {}
