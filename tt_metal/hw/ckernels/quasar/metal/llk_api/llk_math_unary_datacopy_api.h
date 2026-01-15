// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 *
 * @brief Initialize eltwise unary datacopy operations
 *
 * @tparam DATA_COPY_TYPE sets which src register to copy from, values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @param operand: The input operand circular buffer
 * This function prepares the math hardware to copy a specified number of rows
 * from the srcA or srcB register to the destination register.
 */
template <DataCopyType type, bool IS_32b_DEST_EN>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, IS_32b_DEST_EN>(
        num_faces * face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/**
 *
 * @brief Perform an eltwise unary datacopy operation
 *
 * @param dst_index: Tile index into the destination register
 * @param operand: The input operand circular buffer
 *
 * This function copies a specified number of rows
 * from the srcA or srcB register to the destination register.
 */
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
}
