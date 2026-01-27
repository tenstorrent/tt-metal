// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_mul_reduce_scalar.h"

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Fused multiply and scalar reduction
 *************************************************************************/

// Initialize eltwise multiply for mul_reduce_scalar
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_mul_reduce_scalar_eltwise_init(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>(
        num_faces, acc_to_dest);
}

// Perform eltwise multiply for mul_reduce_scalar
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_mul_reduce_scalar_eltwise(uint dst_index, const bool clear_fp32_dst_acc = true) {
    const std::uint32_t num_faces = 4;

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        NUM_FIDELITY_PHASES,
        binary_reuse_dest>(num_faces, dst_index, clear_fp32_dst_acc);
}

// Initialize reduce for mul_reduce_scalar
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_reduce_init() {
    _llk_math_mul_reduce_scalar_init_<type, dim, is_fp32_dest_acc_en, num_fidelity_phases, enforce_fp32_accumulation>();
}

// Perform column reduction for mul_reduce_scalar (accumulates across tiles)
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false,
    bool scalar = false>
inline void llk_math_mul_reduce_scalar_column(const uint dst_index, const uint num_faces = 4) {
    _llk_math_mul_reduce_scalar_column_<
        type,
        dim,
        is_fp32_dest_acc_en,
        num_fidelity_phases,
        is_int_fpu_en,
        enforce_fp32_accumulation,
        scalar>(dst_index, false, num_faces);
}

// Perform final scalar reduction for mul_reduce_scalar
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_final(const uint dst_index, const uint num_faces = 4) {
    _llk_math_mul_reduce_scalar_final_<
        type,
        dim,
        is_fp32_dest_acc_en,
        num_fidelity_phases,
        is_int_fpu_en,
        enforce_fp32_accumulation,
        false>(dst_index, false, num_faces);
}

// Clear data valid flags after mul_reduce_scalar
inline void llk_math_mul_reduce_scalar_clear_dvalid() { _llk_math_mul_reduce_scalar_clear_dvalid_(); }

// Move destination to source registers for mul_reduce_scalar
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_mul_reduce_scalar_move_dest_to_src(uint32_t idst = 0) {
    _llk_math_mul_reduce_scalar_move_dest_to_src_<binary_reuse_dest>(idst);
}
