// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_matmul_api.h"  // legacy CB-id API + unified llk_math_matmul_init_impl; execute reused as-is
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK MATMUL MATH -- LLKOperand (id-free, compile-time NTTP) overload (init only)
 *
 * Only the init needs an id-free overload: it derives the per-source tile r/c dims + partial_face from the
 * two descriptors (IN0 -> SrcB, IN1 -> SrcA). Matmul math is FORMAT-FREE. The matmul math EXECUTE
 * (llk_math_matmul<fidelity, throttle>(idst, ct, rt)) takes no operand id, so the id-free path reuses the
 * legacy execute directly -- no overload here.
 *
 * partial_face is derived as (in0 total_row_dim() < FACE_R_DIM), exactly as the legacy CB-id
 * llk_math_matmul_init derives it from unpack_tile_r_dim.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor IN0_DESC,
    ckernel::experimental::LLKMemDescriptor IN1_DESC,
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t transpose = 0, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    constexpr std::uint32_t in0_tile_r_dim = IN0_DESC.shape.total_row_dim();
    constexpr std::uint32_t in0_tile_c_dim = IN0_DESC.shape.total_col_dim();
    constexpr std::uint32_t in1_tile_r_dim = IN1_DESC.shape.total_row_dim();
    constexpr std::uint32_t in1_tile_c_dim = IN1_DESC.shape.total_col_dim();
    // Shared derivation (internal/llk_descriptor.h) so MATH and UNPACK agree; == legacy (in0 tile_r < FACE_R_DIM).
    constexpr bool partial_face = ckernel::experimental::matmul_partial_face(IN0_DESC);

    llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}
