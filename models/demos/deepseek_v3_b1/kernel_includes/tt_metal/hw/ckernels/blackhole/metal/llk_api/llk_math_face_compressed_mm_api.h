// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_face_compressed_mm.h"
#include "llk_math_common_api.h"

/*************************************************************************
 * LLK MATH FACE_COMPRESSED_MM
 *
 * Face-granular (16x16) variant of compressed_custom_mm. operand0 is the activation
 * (goes to SrcB); the BFP-compressed B faces are addressed from the meta buffer, not a
 * CB, so — unlike the tile version — there is no operand1 base-address read and ct_dim
 * is a compile-time template. The (operand0, operand1) pair is kept uniform across the
 * entry points for a normalized API; operand1 (the compressed-B CB) is unused on this
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

template <std::uint32_t ct_dim = 1>
inline void llk_math_face_compressed_mm_init(const std::uint32_t operand0, const std::uint32_t operand1) {
    // operand0 is the activation (SrcB); operand1 (compressed-B) is unused here.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_face_compressed_mm_init_<ct_dim>(operandB_face_r_dim);
}

template <std::uint32_t ct_dim = 1, bool finalize = true>
inline void llk_math_face_compressed_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim) {
    // operand0 is the activation (SrcB); operand1 (compressed-B) is unused here.
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_face_compressed_mm_<ct_dim, finalize>(base_address_meta, operandB_face_r_dim, dst_index, kt_dim);
}
