// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_math_custom_mm.h"
#include "llk_math_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATH CUSTOM_MM
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32], or [32, 16] without transpose
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16
 * kt_dim: any integer from 1 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_math_custom_mm.h as the low-level implementation.
 *************************************************************************/

// in1 must be [32, 32], or [32, 16] without transpose. Reads the CB's 2D face grid, since a flat face count of 2
// cannot tell [32, 16] (2x1 faces) from [16, 32] (1x2 faces).
template <bool transpose = false>
inline std::uint32_t llk_math_custom_mm_in1_num_faces(const std::uint32_t operand_id) {
    const ckernel::TensorShape in1_shape = get_operand_tensor_shape(operand_id);
    LLK_ASSERT(
        in1_shape.face_r_dim == FACE_R_DIM && in1_shape.num_faces_r_dim == 2 &&
            (in1_shape.num_faces_c_dim == 2 || (in1_shape.num_faces_c_dim == 1 && !transpose)),
        "custom_mm: in1 tile shape must be [32, 32], or [32, 16] without transpose");
    return in1_shape.num_faces_c_dim == 1 ? 2 : TILE_NUM_FACES;
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
inline void llk_math_custom_mm_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const std::uint32_t operandA_num_faces = llk_math_custom_mm_in1_num_faces<transpose>(get_operand_id(operand1));

    _llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(operandB_face_r_dim, ct_dim, operandA_num_faces);
}

template <bool finalize = true>
inline void llk_math_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const std::uint32_t operandA_num_faces = llk_math_custom_mm_in1_num_faces(get_operand_id(operand1));

    _llk_math_custom_mm_<finalize>(operandB_face_r_dim, dst_index, kt_dim, ct_dim, operandA_num_faces);
}
