// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_binary_api.h"
#include "llk_math_common_api.h"
#include "experimental/llk_math_mul_reduce_scalar.h"

/**
 * @brief Init the multiply phase, an ELWMUL of operand_A and operand_B into DEST.
 *
 * @tparam math_fidelity: Fidelity of the multiply.
 * @param operand_A: DFB id of the first input.
 * @param acc_to_dest: Accumulate the product into DEST instead of overwriting it.
 * @param operand_B: DFB id of the second input. Quasar needs its format.
 */
template <MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar_init(
    const std::uint32_t operand_A, const std::uint32_t acc_to_dest, const std::uint32_t operand_B) {
    llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, math_fidelity>(
        operand_A, operand_B, acc_to_dest != 0);
}

/**
 * @brief Multiply the current SrcA and SrcB tiles into DEST tile dst_index.
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void llk_math_eltwise_mul_reduce_scalar(
    const std::uint32_t dst_index, [[maybe_unused]] const std::uint32_t icb0, const bool clear_fp32_dst_acc = true) {
    llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, is_fp32_dest_acc_en, math_fidelity>(
        dst_index, clear_fp32_dst_acc);
}

// The reduce tail calls take the DFB ids of both inputs. operand_A is the reduced operand in SrcA and
// sets the tile shape. operand_B is the operand whose format SrcB holds. A sum passes its input twice.

/**
 * @brief Format the unpacker hands to the source register for a DFB.
 */
inline DataFormat llk_math_mul_reduce_scalar_src_format(const std::uint32_t operand) {
    return static_cast<DataFormat>(get_operand_dst_format(get_operand_id(operand)));
}

/**
 * @brief Init the reduce tail.
 *
 * @param operand_A: DFB id of the reduced input, which sets the SrcA format and the tile shape.
 * @param operand_B: DFB id of the input whose format SrcB holds.
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void llk_math_mul_reduce_scalar_reduce_init(const std::uint32_t operand_A, const std::uint32_t operand_B) {
    static_assert(!enforce_fp32_accumulation || is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    _llk_math_mul_reduce_scalar_init_<math_fidelity>(
        llk_math_mul_reduce_scalar_src_format(operand_A),
        llk_math_mul_reduce_scalar_src_format(operand_B),
        get_operand_tensor_shape(get_operand_id(operand_A)));
}

/**
 * @brief Add the column sums of the tile in SrcA to DEST tile dst_index.
 */
template <MathFidelity math_fidelity>
inline void llk_math_mul_reduce_column(
    const std::uint32_t dst_index, const std::uint32_t operand_A, const std::uint32_t operand_B) {
    _llk_math_mul_reduce_column_<math_fidelity>(
        dst_index,
        llk_math_mul_reduce_scalar_src_format(operand_A),
        llk_math_mul_reduce_scalar_src_format(operand_B),
        get_operand_tensor_shape(get_operand_id(operand_A)));
}

/**
 * @brief Sum the column sums in DEST tile 0 into datum 0.
 */
template <MathFidelity math_fidelity, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
inline void llk_math_mul_reduce_scalar(const std::uint32_t operand_A, const std::uint32_t operand_B) {
    _llk_math_mul_reduce_scalar_<math_fidelity, is_fp32_dest_acc_en>(
        llk_math_mul_reduce_scalar_src_format(operand_A), llk_math_mul_reduce_scalar_src_format(operand_B));
}

/**
 * @brief Release the source banks held by the reduce tail.
 */
inline void llk_math_mul_reduce_scalar_clear_dvalid() { _llk_math_mul_reduce_scalar_clear_dvalid_(); }

/**
 * @brief Move DEST tile idst into SrcA, or its row 0 into SrcB.
 *
 * @param operand_A: DFB id of the reduced input, which sets the tile shape.
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void llk_math_mul_reduce_scalar_move_dest_to_src(const std::uint32_t idst, const std::uint32_t operand_A) {
    _llk_math_mul_reduce_scalar_move_dest_to_src_<binary_reuse_dest>(
        idst, get_operand_tensor_shape(get_operand_id(operand_A)));
}
