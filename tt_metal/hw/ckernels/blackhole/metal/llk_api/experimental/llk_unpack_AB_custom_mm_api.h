// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB CUSTOM_MM
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16
 * kt_dim: any integer from 1 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_unpack_AB_custom_mm.h as the low-level implementation.
 *************************************************************************/

/**
 * @brief Configure the unpack thread for a custom_mm block matmul.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>
 * @param operand0: CB of the activations, whose face_r_dim this reads. Its data goes to SrcB.
 * @param operand1: CB of the weights, whose unpack format selects the instruction tuning. Its data goes to SrcA.
 * @param ct_dim: Output width in tiles, 1 to 16.
 * @note Call this before @ref llk_unpack_AB_custom_mm, and again after any other op has run, in particular one
 *       that writes SrcB.
 * @note On the math thread, pair with @ref llk_math_custom_mm_init.
 */
template <bool transpose = false, bool clear_src = true>
inline void llk_unpack_AB_custom_mm_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
    const std::uint32_t operandA_unpack_dst_format = unpack_dst_format[operandA_id];

    _llk_unpack_AB_custom_mm_init_<transpose, clear_src>(operandB_face_r_dim, operandA_unpack_dst_format, ct_dim);
}

/**
 * @brief Unpack a kt_dim x ct_dim block: weight tiles from operand1 into SrcA, activation tiles from operand0 into
 *        SrcB.
 *
 * @tparam read_transposed: Walk the weight tiles column by column instead of row by row, values = <true/false>
 * @param operand0: CB of the activations; its read pointer is the SrcB base.
 * @param operand1: CB of the weights; its read pointer is the SrcA base.
 * @param tile_index_0: First activation tile, relative to operand0's read pointer.
 * @param tile_index_1: First weight tile, relative to operand1's read pointer.
 * @param kt_dim: Inner dimension in tiles, 1 to 256.
 * @param ct_dim: Output width in tiles, 1 to 16.
 * @note Call @ref llk_unpack_AB_custom_mm_init first.
 * @note On the math thread, pair with @ref llk_math_custom_mm.
 */
template <bool read_transposed = false>
inline void llk_unpack_AB_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t tile_index_0,
    const std::uint32_t tile_index_1,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    const std::uint32_t tile_index_A = tile_index_1;
    const std::uint32_t tile_index_B = tile_index_0;
    const std::uint32_t tile_size_A = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t tile_size_B = get_local_cb_interface(operandB_id).fifo_page_size;
    _llk_unpack_AB_custom_mm_<read_transposed>(
        base_address_A, base_address_B, tile_index_A, tile_index_B, tile_size_A, tile_size_B, kt_dim, ct_dim);
}
