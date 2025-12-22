// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

template <DataCopyType type, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en>(
        num_faces * face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
}
