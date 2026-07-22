// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "experimental/llk_math_generalized_moe_gate_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with operands
template <EltwiseBinaryType eltwise_binary_type, GeneralizedMoeGateEltwiseBinaryMode mode, MathFidelity math_fidelity>
inline void llk_math_generalized_moe_gate_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_generalized_moe_gate_eltwise_binary_init_<eltwise_binary_type, mode, math_fidelity>(
        num_faces, acc_to_dest);
}

template <EltwiseBinaryType eltwise_binary_type, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_generalized_moe_gate_eltwise_binary(
    const std::uint32_t operand_A, const std::uint32_t operand_B, std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_generalized_moe_gate_eltwise_binary_<
        eltwise_binary_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity>(num_faces, dst_index);
}
