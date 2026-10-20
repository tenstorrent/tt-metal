// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_AB_matmul_api.h"  // legacy CB-id API + unified llk_unpack_AB_matmul_{init_,}impl
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API, distinguished by two LLKMemDescriptor NTTPs: IN0_DESC (-> SrcB) and
 * IN1_DESC (-> SrcA). Matmul unpack is FORMAT-FREE at the op level (formats set at
 * compute_kernel_hw_startup<SrcOrder::Reverse>), so these forward only geometry + addresses + per-tile sizes.
 *
 * ROLE SWAP (applied by the LLK): in0 -> SrcB, in1 -> SrcA. Both init and execute take the IN0 and IN1
 * shapes; base_a/tile_size_a come from IN0, base_b/tile_size_b from IN1.
 *
 * ASSUMPTIONS (documented at the compute layer, 2_0/matmul.h):
 *   - the LLK derives the unpack partial_face from each shape as (total_row_dim() < TILE_R_DIM), the same rule
 *     as the host-side Tile::partial_face. This is a looser threshold than the MATH engine's rule
 *     (llk_math_matmul.h uses in0 face_r_dim < FACE_R_DIM); see llk_descriptor.h.
 *   - per-tile size derived from the descriptor (fifo_page_size == a single tile's size; exact for linear
 *     formats, single-tile test path never applies the multiplier).
 *************************************************************************/

template <ckernel::experimental::LLKMemDescriptor IN0_DESC, ckernel::experimental::LLKMemDescriptor IN1_DESC>
inline void llk_unpack_AB_matmul_init(
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    SAN_HOOK(unsupported());
    // SrcB <- IN0, SrcA <- IN1 (applied by the LLK).
    llk_unpack_AB_matmul_init_impl(transpose, ct_dim, rt_dim, kt_dim, IN0_DESC.shape, IN1_DESC.shape);
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
    SAN_HOOK(unsupported());
    // base_a/tile_size_a <- IN0, base_b/tile_size_b <- IN1; the IN0/IN1 shapes go to the LLK as
    // in0/in1, which derives face-by-face unpacking from them.
    llk_unpack_AB_matmul_impl(
        base_ptr_in0,
        base_ptr_in1,
        tile_index_in0,
        tile_index_in1,
        ckernel::experimental::tile_stride_words(IN0_DESC.format, IN0_DESC.shape),
        ckernel::experimental::tile_stride_words(IN1_DESC.format, IN1_DESC.shape),
        IN0_DESC.shape,
        IN1_DESC.shape,
        ct_dim,
        rt_dim,
        kt_dim);
}
