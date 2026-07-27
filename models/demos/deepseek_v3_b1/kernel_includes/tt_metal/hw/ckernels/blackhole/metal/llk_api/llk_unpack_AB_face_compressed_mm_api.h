// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_face_compressed_mm.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB FACE_COMPRESSED_MM
 *
 * Face-granular (16x16) variant of compressed_custom_mm. operand0 is the activation
 * (goes to SrcB); the BFP-compressed B faces are addressed from the meta buffer, not a
 * CB, so — unlike the tile version — there is no operand1 base-address read and ct_dim
 * is a compile-time template. The (operand0, operand1) pair is kept uniform across the
 * entry points for a normalized API; operand1 (the compressed-B CB) is read only by the
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

template <bool transpose = false>
inline void llk_unpack_AB_face_compressed_mm_init(const std::uint32_t operand0, const std::uint32_t operand1) {
    // operand0 is the activation (goes to SrcB); operand1 (compressed-B) is unused here.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_unpack_AB_face_compressed_mm_init_<transpose>(operandB_face_r_dim);
}

template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
inline void llk_unpack_AB_face_compressed_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim) {
    // operand0 (activation) -> SrcB base address; operand1 (compressed-B) is unused here
    // (the compressed B comes from base_address_meta).
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t base_address_B = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    _llk_unpack_AB_face_compressed_mm_<ct_dim, clear_src, finalize>(base_address_B, base_address_meta, kt_dim);
}

inline void llk_unpack_AB_face_compressed_mm_uninit(const std::uint32_t operand0, const std::uint32_t operand1) {
    // operand0 (activation) is unused here; restore the compressed-B (SrcA) tile
    // descriptor num_faces from operand1 that the init forced to a single face.
    const std::uint32_t operandA_id = get_operand_id(operand1);
    const std::uint32_t operandA_num_faces = get_operand_num_faces(operandA_id);

    _llk_unpack_AB_face_compressed_mm_uninit_(operandA_num_faces);
}
