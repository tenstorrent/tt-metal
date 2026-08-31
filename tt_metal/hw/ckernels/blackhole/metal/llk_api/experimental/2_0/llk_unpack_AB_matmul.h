// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_AB_matmul_api.h"  // legacy CB-id API + unified llk_unpack_AB_matmul_{init_,}impl
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API, distinguished by two LLKMemDescriptor NTTPs: IN0_DESC (-> SrcB) and
 * IN1_DESC (-> SrcA). Matmul unpack is FORMAT-FREE at the op level (formats set at
 * compute_kernel_hw_startup<SrcOrder::Reverse>), so these forward only geometry + addresses + per-tile sizes.
 *
 * ROLE SWAP (preserved from legacy): in0 -> SrcB, in1 -> SrcA.
 *   - init: "unpA"(SrcA) geometry comes from IN1, "unpB"(SrcB) from IN0.
 *   - execute: base_a/tile_size_a from IN0, base_b/tile_size_b from IN1, but partial_face_a from IN1 and
 *     partial_face_b from IN0 (mirrors the legacy execute's operand wiring exactly, incl. its known quirk).
 *
 * ASSUMPTIONS (documented at the compute layer, 2_0/matmul.h):
 *   - partial_face is derived as (total_row_dim() < FACE_R_DIM) -- the same rule MATH already uses in
 *     llk_math_matmul_init; equals the host unpack_partial_face[] for the tested full tiles.
 *   - per-tile size derived from the descriptor (fifo_page_size == a single tile's size; exact for linear
 *     formats, single-tile test path never applies the multiplier).
 *************************************************************************/

// matmul_partial_face / matmul_tile_size are shared helpers in internal/llk_descriptor.h (used by both the
// matmul unpack here and the matmul math), so they are derived identically in one place.

template <ckernel::experimental::LLKMemDescriptor IN0_DESC, ckernel::experimental::LLKMemDescriptor IN1_DESC>
inline void llk_unpack_AB_matmul_init(
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // unpA(SrcA) <- IN1, unpB(SrcB) <- IN0.
    llk_unpack_AB_matmul_init_impl(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim,
        IN1_DESC.shape.face_r_dim,
        IN0_DESC.shape.face_r_dim,
        IN1_DESC.shape.total_num_faces(),
        IN0_DESC.shape.total_num_faces(),
        ckernel::experimental::matmul_partial_face(IN1_DESC),
        ckernel::experimental::matmul_partial_face(IN0_DESC));
}

template <ckernel::experimental::LLKMemDescriptor IN0_DESC, ckernel::experimental::LLKMemDescriptor IN1_DESC>
inline void llk_unpack_AB_matmul(
    std::uint32_t base_ptr_in0,
    std::uint32_t base_ptr_in1,
    std::uint32_t tile_index_in0,
    std::uint32_t tile_index_in1,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // Legacy execute wiring: base_a/tile_size_a <- IN0, base_b/tile_size_b <- IN1; partial_face_a <- IN1,
    // partial_face_b <- IN0 (preserved verbatim, including the "TODO: Review RT" quirk).
    llk_unpack_AB_matmul_impl(
        base_ptr_in0,
        base_ptr_in1,
        tile_index_in0,
        tile_index_in1,
        ckernel::experimental::matmul_tile_size(IN0_DESC),
        ckernel::experimental::matmul_tile_size(IN1_DESC),
        ckernel::experimental::matmul_partial_face(IN1_DESC),
        ckernel::experimental::matmul_partial_face(IN0_DESC),
        ct_dim,
        rt_dim,
        kt_dim);
}
