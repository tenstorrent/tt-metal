// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_common_api.h"
#include "experimental/llk_math_matmul_custom_no_mop.h"

/*************************************************************************
 * LLK MATMUL (NO MOP)
 *************************************************************************/

// Both operands must agree on 2x-ness, as llk_math_matmul_init also requires.
// Re-derived per call rather than cached at init so the execute path stays stateless (the MOP path
// keeps this implicitly in its MOP config, which the no-MOP path has by definition given up).
inline bool operands_use_2x_format(const std::uint32_t operand0, const std::uint32_t operand1) {
    const DataFormat format0 = static_cast<DataFormat>(get_operand_dst_format(get_operand_id(operand0)));
    const DataFormat format1 = static_cast<DataFormat>(get_operand_dst_format(get_operand_id(operand1)));
    LLK_ASSERT(
        is_2x_format(format0) == is_2x_format(format1), "Both operands must be 2x formats or both non-2x formats");
    return is_2x_format(format0) && is_2x_format(format1);
}

/**
 * @brief Initialize a matrix multiply of Input 0 * Input 1 -> SrcB * SrcA that runs without a MOP.
 *
 * Configures the ALU data-format state, then programs the matmul addrmods and records the replay buffer.
 * MOP BANK0 is left untouched, so a fused op may own it.
 *
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 controls precision of multiplication
 * @tparam THROTTLE_LEVEL: Accepted for API parity with Wormhole/Blackhole; Quasar has no throttled MVMUL sequences, so
 * only 0 is valid
 * @param operandA: Logical dataflow buffer identifier for input 0 (-> SrcB)
 * @param operandB: Logical dataflow buffer identifier for input 1 (-> SrcA)
 * @param transpose: Transpose flag; not supported on Quasar. Present so this signature matches
 *        the Wormhole/Blackhole llk_api and the shared Compute API needs no arch branch.
 * @param ct_dim: number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: number of tiles in the row dimension for a matrix multiply
 * @note Run @ref llk_math_matmul_no_mop with matching template args to execute the configured matmul.
 * @note Run @ref llk_math_matmul_reinit_no_mop instead when only the replay buffer and addrmods need
 *       restoring, e.g. after an interleaved op that recorded over replay buffer slot 0.
 */
template <ckernel::MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    static_assert(
        THROTTLE_LEVEL == 0,
        "Quasar no-mop matmul only supports THROTTLE_LEVEL == 0; Quasar has no throttled MVMUL sequences");
    LLK_ASSERT(!transpose, "non-default transpose not supported on Quasar");

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // The replayed MVMUL walk marches SrcA/SrcB across all four faces, so it is valid for full 32x32
    // tiles only: the same restriction llk_unpack_AB_matmul_init enforces on the unpack side.
    LLK_ASSERT(
        get_operand_tensor_shape(operandA_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "no-mop matmul replays a four-face MVMUL walk, so it supports full 32x32 tiles only");
    LLK_ASSERT(
        get_operand_tensor_shape(operandB_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "no-mop matmul replays a four-face MVMUL walk, so it supports full 32x32 tiles only");

    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operandA_id));
    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operandB_id));
    _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, DST_ACCUM_MODE>(
        srcA_format, srcB_format);

    if (operands_use_2x_format(operandA, operandB)) {
        _llk_math_matmul_init_no_mop_<math_fidelity, true /*EN_X2*/>(ct_dim, rt_dim);
    } else {
        _llk_math_matmul_init_no_mop_<math_fidelity, false /*EN_X2*/>(ct_dim, rt_dim);
    }
}

/**
 * @brief Performs a matrix multiply of Input 0 * Input 1 -> SrcB * SrcA over a block of tiles, without a MOP.
 *
 * Input 0 dim = [rt_dim, 1] -> SrcB reg, Input 1 dim = [1, ct_dim] -> SrcA reg;
 * output is a matrix block of dimension [rt_dim, ct_dim].
 * This function does not iterate over kt_dim, must iterate over kt_dim externally to this function.
 * Dest index is always assumed to start at 0 for this operation.
 *
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 * @tparam THROTTLE_LEVEL: Accepted for API parity with Wormhole/Blackhole; Quasar has no throttled MVMUL sequences, so
 * only 0 is valid
 * @param operandA: Logical dataflow buffer identifier for input 0 (-> SrcB)
 * @param operandB: Logical dataflow buffer identifier for input 1 (-> SrcA)
 * @param dst_index: First destination tile index; only 0 is supported on Quasar, whose block matmul
 *        always starts at dest tile 0. Present so this signature matches the Wormhole/Blackhole
 *        llk_api and the shared Compute API needs no arch branch.
 * @param ct_dim: number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: number of tiles in the row dimension for a matrix multiply
 * @note Run @ref llk_math_matmul_init_no_mop with matching template args first.
 */
template <ckernel::MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t dst_index,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    static_assert(
        THROTTLE_LEVEL == 0,
        "Quasar no-mop matmul only supports THROTTLE_LEVEL == 0; Quasar has no throttled MVMUL sequences");
    LLK_ASSERT(dst_index == 0, "non-default dst_index not supported on Quasar");

    // Re-derive 2x-ness so the execute issues the same MVMUL count that init recorded.
    if (operands_use_2x_format(operandA, operandB)) {
        _llk_math_matmul_block_no_mop_<math_fidelity, true /*EN_X2*/>(ct_dim, rt_dim);
    } else {
        _llk_math_matmul_block_no_mop_<math_fidelity, false /*EN_X2*/>(ct_dim, rt_dim);
    }
}

/**
 * @brief Restores the no-mop matmul math state without touching the ALU data-format state.
 *
 * Reprograms the matmul addrmods and re-records the replay buffer. Unlike Wormhole/Blackhole where the
 * math thread owns a dedicated replay-buffer offset and a reinit only has to restore addrmods every
 * Quasar LLK records at replay buffer slot 0, so an interleaved op overwrites this matmul's image and it
 * must be re-recorded here.
 *
 * @tparam math_fidelity: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication
 * @tparam THROTTLE_LEVEL: Accepted for API parity with Wormhole/Blackhole; Quasar has no throttled MVMUL sequences, so
 * only 0 is valid
 * @param operandA: Logical dataflow buffer identifier for input 0 (-> SrcB)
 * @param operandB: Logical dataflow buffer identifier for input 1 (-> SrcA)
 * @param transpose: Transpose flag; only false is supported on Quasar. Present so this signature matches
 *        the Wormhole/Blackhole llk_api and the shared Compute API needs no arch branch.
 * @param ct_dim: number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: number of tiles in the row dimension for a matrix multiply
 * @note Run @ref llk_math_matmul_init_no_mop once before the steady-state loop; this call is the
 *       per-iteration restore.
 */
template <ckernel::MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_reinit_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    static_assert(
        THROTTLE_LEVEL == 0,
        "Quasar no-mop matmul only supports THROTTLE_LEVEL == 0; Quasar has no throttled MVMUL sequences");
    LLK_ASSERT(!transpose, "non-default transpose not supported on Quasar");

    if (operands_use_2x_format(operandA, operandB)) {
        _llk_math_matmul_init_no_mop_<math_fidelity, true /*EN_X2*/>(ct_dim, rt_dim);
    } else {
        _llk_math_matmul_init_no_mop_<math_fidelity, false /*EN_X2*/>(ct_dim, rt_dim);
    }
}
