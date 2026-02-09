// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "../../../../../../tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_deepseek_moe_gate_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with operands
template <EltwiseBinaryType eltwise_binary_type, DeepseekMoeGateEltwiseBinaryMode mode, int NUM_FIDELITY_PHASES = 0>
inline void llk_math_deepseek_moe_gate_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_deepseek_moe_gate_eltwise_binary_init_<eltwise_binary_type, mode, NUM_FIDELITY_PHASES>(
        num_faces, acc_to_dest);
}

template <EltwiseBinaryType eltwise_binary_type, bool is_fp32_dest_acc_en, int NUM_FIDELITY_PHASES = 0>
inline void llk_math_deepseek_moe_gate_eltwise_binary(
    const std::uint32_t operand_A, const std::uint32_t operand_B, uint dst_index, const bool clear_fp32_dst_acc) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_deepseek_moe_gate_eltwise_binary_<
        eltwise_binary_type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        NUM_FIDELITY_PHASES>(num_faces, dst_index, clear_fp32_dst_acc);
}
