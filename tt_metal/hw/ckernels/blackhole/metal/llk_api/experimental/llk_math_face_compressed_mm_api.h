// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "experimental/llk_math_face_compressed_mm.h"
#include "llk_math_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATH FACE_COMPRESSED_MM
 *
 * Face-granular (16x16) variant of compressed_custom_mm. operand0 is the activation
 * (goes to SrcB); the BFP-compressed weight faces (SrcA) are addressed from the meta buffer,
 * not a CB, so — unlike the tile version — there is no operand1 base-address read and ct_dim
 * is a compile-time template. The (operand0, operand1) pair is kept uniform across the
 * entry points for a normalized API; operand1 (the compressed-weight CB) is unused on this
 * thread (only operand0's SrcB face_r_dim is read).
 * Limits:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_math_face_compressed_mm.h as the low-level implementation.
 *************************************************************************/

/**
 * @brief Configure the math thread for a face-granular compressed matmul.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param operand0: CB of the activation, whose face_r_dim this reads.
 * @param operand1: CB of the compressed weights, unused on this thread.
 * @note Call this before @ref llk_math_face_compressed_mm with the same ct_dim. This thread has no uninit.
 * @note On the unpack thread, pair with @ref llk_unpack_AB_face_compressed_mm_init.
 */
template <std::uint32_t ct_dim = 1>
inline void llk_math_face_compressed_mm_init(
    const std::uint32_t operand0, [[maybe_unused]] const std::uint32_t operand1) {
    SAN_HOOK(unsupported());
    // Only operand0's SrcB face_r_dim is read; operand1 is the compressed-weight CB.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_face_compressed_mm_init_<ct_dim>(operandB_face_r_dim);
}

/**
 * @brief Multiply the unpacked activation by the compressed weight faces, accumulating into DST.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam finalize: Merge the split-accumulation partials, values = <true/false>. Only ct_dim == 1 splits.
 * @param operand0: CB of the activation, whose face_r_dim this reads.
 * @param operand1: CB of the compressed weights, unused on this thread.
 * @param base_address_meta: L1 address of the meta buffer, whose first section holds the math metas.
 * @param dst_index: Tile index in DST that the result is written to.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref llk_math_face_compressed_mm_init first, with the same ct_dim.
 * @note On the unpack thread, pair with @ref llk_unpack_AB_face_compressed_mm.
 */
template <std::uint32_t ct_dim = 1, bool finalize = true>
inline void llk_math_face_compressed_mm(
    const std::uint32_t operand0,
    [[maybe_unused]] const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim) {
    SAN_HOOK(unsupported());
    // Only operand0's SrcB face_r_dim is read; the compressed weights come from base_address_meta.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_face_compressed_mm_<ct_dim, finalize>(base_address_meta, operandB_face_r_dim, dst_index, kt_dim);
}
