// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with no operand (assumes default 32x32 tile)
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0) {
    _llk_math_eltwise_binary_init_<
        eltwise_binary_type,
        src_b_bcast_type,
        math_fidelity,
        binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE, acc_to_dest);
}

// Version with operands
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_eltwise_binary_init_<
        eltwise_binary_type,
        src_b_bcast_type,
        math_fidelity,
        binary_reuse_dest>(tensor_shape, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = true) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE, dst_index, clear_fp32_dst_acc);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    uint dst_index,
    const bool clear_fp32_dst_acc = true) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(tensor_shape, dst_index, clear_fp32_dst_acc);
}
