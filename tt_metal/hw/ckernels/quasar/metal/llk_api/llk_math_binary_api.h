// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with no operand (assumes default 32x32 tile)
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool enable_direct_indexing = false>
inline void llk_math_eltwise_binary_init([[maybe_unused]] const std::uint32_t acc_to_dest = 0) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that acc_to_dest is unused
    _llk_math_eltwise_binary_init_<eltwise_binary_type, math_fidelity, enable_direct_indexing, binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE);
}

// Version with operands
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool enable_direct_indexing = false>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, [[maybe_unused]] const std::uint32_t operand_B, [[maybe_unused]] const std::uint32_t acc_to_dest = 0) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that acc_to_dest is unused
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape_A = get_operand_tensor_shape(operand_id);

    _llk_math_eltwise_binary_init_<eltwise_binary_type, math_fidelity, enable_direct_indexing, binary_reuse_dest>(tensor_shape_A);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(uint dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc = true) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that clear_fp32_dst_acc is unused
    WAYPOINT("MBIW");
    _llk_math_eltwise_binary_<binary_reuse_dest>(dst_index);
    WAYPOINT("MBID");
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A, [[maybe_unused]] const std::uint32_t operand_B, uint dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc) {
    static_assert(src_b_bcast_type == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that clear_fp32_dst_acc is unused
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    WAYPOINT("MBIW");
    _llk_math_eltwise_binary_<binary_reuse_dest>(dst_index, num_faces);
    WAYPOINT("MBID");
}
