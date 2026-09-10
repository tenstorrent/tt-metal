// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "experimental/llk_unpack_AB_face_compressed_mm.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB FACE_COMPRESSED_MM
 *
 * Face-granular (16x16) variant of compressed_custom_mm. operand0 is the activation
 * (goes to SrcB); the BFP-compressed weight faces (SrcA) are addressed from the meta buffer,
 * not a CB, so — unlike the tile version — there is no operand1 base-address read and ct_dim
 * is a compile-time template. The (operand0, operand1) pair is kept uniform across the
 * entry points for a normalized API; operand1 (the compressed-weight CB) is read only by the
 * uninit, to restore the tile descriptor the init forced to a single face.
 * Limits:
 * in0 tile shape: [{1, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 16 (compile-time)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_unpack_AB_face_compressed_mm.h as the low-level implementation.
 *************************************************************************/

/**
 * @brief Configure the unpack thread for a face-granular compressed matmul.
 *
 * @tparam transpose: Haloize the SrcA read, values = <true/false>
 * @param operand0: CB of the activation, whose face_r_dim this reads. Its data goes to SrcB.
 * @param operand1: CB of the compressed weights, read only by
 *                  @ref llk_unpack_AB_face_compressed_mm_uninit.
 * @note Call this before @ref llk_unpack_AB_face_compressed_mm, and
 *       @ref llk_unpack_AB_face_compressed_mm_uninit after the last one.
 * @note On the math thread, pair with @ref llk_math_face_compressed_mm_init.
 */
template <bool transpose = false>
inline void llk_unpack_AB_face_compressed_mm_init(
    const std::uint32_t operand0, [[maybe_unused]] const std::uint32_t operand1) {
    SAN_HOOK(unsupported());
    // operand0 is the activation, which goes to SrcB; operand1 is the compressed-weight CB.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_unpack_AB_face_compressed_mm_init_<transpose>(operandB_face_r_dim);
}

/**
 * @brief Unpack the activation block into SrcB and the compressed weight faces into SrcA.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam clear_src: Clear SrcB before the first unpack, values = <true/false>
 * @tparam finalize: For ct_dim == 1, leave both sources zeroed and valid so the math thread can merge its
 *                   split-accumulation partials, values = <true/false>
 * @param operand0: CB of the activation; its read pointer becomes the SrcB base address.
 * @param operand1: CB of the compressed weights, unused here -- they are addressed from base_address_meta.
 * @param base_address_meta: L1 address of the meta buffer, holding the math metas, the per-chunk weight
 *                           base addresses and the unpack index words.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref llk_unpack_AB_face_compressed_mm_init first.
 * @note On the math thread, pair with @ref llk_math_face_compressed_mm.
 */
template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
inline void llk_unpack_AB_face_compressed_mm(
    const std::uint32_t operand0,
    [[maybe_unused]] const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim) {
    SAN_HOOK(unsupported());
    // operand0, the activation, supplies the SrcB base address. The compressed weights are addressed
    // from base_address_meta, so operand1's CB is never read here.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    _llk_unpack_AB_face_compressed_mm_<ct_dim, clear_src, finalize>(base_address_B, base_address_meta, kt_dim);
}

/**
 * @brief Restore the unpacker state the init changed.
 *
 * @param operand0: CB of the activation, unused here.
 * @param operand1: CB of the compressed weights, whose tile descriptor num_faces this puts back.
 * @note Call after the last @ref llk_unpack_AB_face_compressed_mm.
 */
inline void llk_unpack_AB_face_compressed_mm_uninit(
    [[maybe_unused]] const std::uint32_t operand0, const std::uint32_t operand1) {
    SAN_HOOK(unsupported());
    // Restore the compressed-weight (SrcA) tile descriptor num_faces from operand1, which the init forced
    // to a single face.
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandA_num_faces = get_operand_num_faces(operandA_id);

    _llk_unpack_AB_face_compressed_mm_uninit_(operandA_num_faces);
}
