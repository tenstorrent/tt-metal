// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_assert.h"
#include "llk_math_common_api.h"

/*************************************************************************
 * LLK MATH ELTWISE BINARY CUSTOM - SDPA specialized blocked sub path
 *************************************************************************/

template <MathFidelity math_fidelity>
inline void llk_math_eltwise_binary_sub_bcast_cols_init_custom(
    const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_custom_<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(num_faces);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sub_bcast_cols_custom(
    const std::uint32_t operandA, const std::uint32_t dst_index, const std::uint32_t ct_dim = 1) {
    LLK_ASSERT(
        (dst_index + ct_dim <= get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "dst_index + ct_dim out of range");

    const std::uint32_t operand_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    // dst_index is the absolute base dest tile slot; the LLK helper writes ct_dim
    // consecutive tiles from there and restores the dest base on exit.
    _llk_math_sub_bcast_cols_reuse_custom_(ct_dim, tensor_shape, dst_index);
}
