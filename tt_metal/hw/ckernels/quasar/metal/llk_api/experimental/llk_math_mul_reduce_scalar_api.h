// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_binary_api.h"
#include "llk_math_common_api.h"
#include "experimental/llk_math_mul_reduce_scalar.h"

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Fused multiply and scalar reduction (Quasar)
 *************************************************************************/

/**
 * @brief Init the multiply phase (ELWMUL of operand_A and operand_B into DEST).
 *
 * @tparam math_fidelity: Fidelity of the multiply.
 * @param operand_A: DFB id of the first input.
 * @param acc_to_dest: Accumulate the product into DEST instead of overwriting it.
 * @param operand_B: DFB id of the second input; Quasar needs its format for the ALU state and fidelity schedule.
 */
template <MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar_init(
    const std::uint32_t operand_A, const std::uint32_t acc_to_dest, const std::uint32_t operand_B) {
    llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, math_fidelity>(
        operand_A, operand_B, acc_to_dest != 0);
}

/**
 * @brief Multiply the current SrcA/SrcB tiles into DEST tile dst_index.
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar(
    const std::uint32_t dst_index, [[maybe_unused]] const std::uint32_t icb0, const bool clear_fp32_dst_acc = true) {
    llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, is_fp32_dest_acc_en, math_fidelity>(
        dst_index, clear_fp32_dst_acc);
}

/**
 * @brief Init the reduce tail.
 *
 * @param operand: DFB id of the reduced input; its format sets the pool fidelity schedule.
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_reduce_init(const std::uint32_t operand) {
    static_assert(!enforce_fp32_accumulation || is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, math_fidelity>(
        static_cast<DataFormat>(get_operand_dst_format(operand_id)));
}

/**
 * @brief Column-pool the tile in SrcA into DEST tile dst_index, accumulating.
 */
template <MathFidelity math_fidelity>
inline void llk_math_mul_reduce_column(const std::uint32_t dst_index, const std::uint32_t icb0) {
    const std::uint32_t operand_id = get_operand_id(icb0);
    _llk_math_mul_reduce_column_(dst_index, get_operand_tensor_shape(operand_id));
}

/**
 * @brief Collapse the column sums in DEST tile 0 into datum 0.
 */
template <MathFidelity math_fidelity, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
inline void llk_math_mul_reduce_scalar() {
    _llk_math_mul_reduce_scalar_<is_fp32_dest_acc_en>();
}

/**
 * @brief Release the source banks held by the reduce tail.
 */
inline void llk_math_mul_reduce_scalar_clear_dvalid() { _llk_math_mul_reduce_scalar_clear_dvalid_(); }

/**
 * @brief Move DEST tile idst into SrcA (DEST_TO_SRCA) or its first face into SrcB (DEST_TO_SRCB).
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void llk_math_mul_reduce_scalar_move_dest_to_src(const std::uint32_t idst = 0) {
    _llk_math_mul_reduce_scalar_move_dest_to_src_<binary_reuse_dest>(idst);
}
