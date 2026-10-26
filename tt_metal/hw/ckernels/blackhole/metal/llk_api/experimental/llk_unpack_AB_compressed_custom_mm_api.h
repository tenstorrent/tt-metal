// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_unpack_AB_compressed_custom_mm.h"
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
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_unpack_AB_custom_mm.h as the low-level implementation.
 *************************************************************************/

/**
 * @brief Configure the unpack thread for a compressed custom_mm block matmul.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>
 * @param operand0: CB of the activations, whose face_r_dim this reads. Its data goes to SrcB.
 * @param operand1: CB of the compressed weights, unused here.
 * @note Call this before @ref llk_unpack_AB_compressed_custom_mm, and again after any other op has run, in
 *       particular one that writes SrcB, SCRATCH_SEC0-2 or GPRs PERF_UNPACK_NUM_TILES_1-3.
 * @note On the math thread, pair with @ref llk_math_compressed_custom_mm_init.
 */
template <bool transpose = false, bool clear_src = true>
inline void llk_unpack_AB_compressed_custom_mm_init(const std::uint32_t operand0, const std::uint32_t operand1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_unpack_AB_compressed_custom_mm_init_<transpose, clear_src>(operandB_face_r_dim);
}

/**
 * @brief Unpack a kt_dim x ct_dim block of compressed weight tiles into SrcA and activation tiles into SrcB.
 *
 * @param operand0: CB of the activations; its read pointer is the SrcB base.
 * @param operand1: CB of the compressed weights; its read pointer is the start of the weight stream.
 * @param base_address_meta: Byte address of the per-tile format metadata; see
 *                           @ref _llk_unpack_AB_compressed_custom_mm_ for the layout.
 * @param kt_dim: Inner dimension in tiles, an even number from 2 to 256.
 * @param ct_dim: Output width in tiles, 1 to 16.
 * @note Call @ref llk_unpack_AB_compressed_custom_mm_init first.
 * @note On the math thread, pair with @ref llk_math_compressed_custom_mm.
 */
inline void llk_unpack_AB_compressed_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_A = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    _llk_unpack_AB_compressed_custom_mm_(base_address_A, base_address_B, base_address_meta, kt_dim, ct_dim);
}
