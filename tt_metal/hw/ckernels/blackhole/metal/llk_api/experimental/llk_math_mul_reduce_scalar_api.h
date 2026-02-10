// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"
#include "experimental/llk_math_mul_reduce_scalar.h"

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Fused multiply and scalar reduction
 *************************************************************************/

template <MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar_init(
    const std::uint32_t operand_A, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_<
        EltwiseBinaryType::ELWMUL,
        BroadcastType::NONE,
        math_fidelity,
        EltwiseBinaryReuseDestType::NONE>(num_faces, acc_to_dest);
}

template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar(
    uint dst_index, const std::uint32_t icb0, const bool clear_fp32_dst_acc = true) {
    const std::uint32_t operand_id = get_operand_id(icb0);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_<
        EltwiseBinaryType::ELWMUL,
        BroadcastType::NONE,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity,
        EltwiseBinaryReuseDestType::NONE>(num_faces, dst_index, clear_fp32_dst_acc);
}

template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_reduce_init() {
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, math_fidelity, enforce_fp32_accumulation>();
}

template <MathFidelity math_fidelity>
inline void llk_math_mul_reduce_column(const uint dst_index, const std::uint32_t icb0) {
    const std::uint32_t operand_id = get_operand_id(icb0);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_math_mul_reduce_column_<math_fidelity>(dst_index, false, num_faces);
}

template <MathFidelity math_fidelity>
inline void llk_math_mul_reduce_scalar() {
    _llk_math_mul_reduce_scalar_<math_fidelity>();
}

inline void llk_math_mul_reduce_scalar_clear_dvalid() { _llk_math_mul_reduce_scalar_clear_dvalid_(); }

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_mul_reduce_scalar_move_dest_to_src(uint32_t idst = 0) {
    _llk_math_mul_reduce_scalar_move_dest_to_src_<binary_reuse_dest>(idst);
}
