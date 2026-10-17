// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_matmul_api.h"  // legacy CB-id API + unified llk_math_matmul_init_impl; execute reused as-is
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATMUL MATH -- LLKOperand (id-free, compile-time NTTP) overload (init only)
 *
 * Only the init needs an id-free overload: it derives the per-source tile r/c dims + partial_face from the
 * two descriptors (IN0 -> SrcB, IN1 -> SrcA). Matmul math is FORMAT-FREE. The matmul math EXECUTE
 * (llk_math_matmul<fidelity, throttle>(idst, ct, rt)) takes no operand id, so the id-free path reuses the
 * legacy execute directly -- no overload here.
 *
 * partial_face is derived INLINE as (in0 total_row_dim() < FACE_R_DIM), exactly as the legacy CB-id
 * llk_math_matmul_init derives it from unpack_tile_r_dim. NOTE this is a DIFFERENT threshold than the UNPACK
 * path (llk_unpack_AB_matmul.h, < TILE_R_DIM); the divergence is inherited from legacy (see llk_descriptor.h).
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor IN0_DESC,
    ckernel::experimental::LLKMemDescriptor IN1_DESC,
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t transpose = 0, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    // Compile-time form of the unified init's runtime LLK_ASSERT (compiled out in production builds), widened to
    // the full support matrix: a pair the dispatch below has no program for would run the 32x32 default over
    // faces the operands do not have.
    static_assert(
        ckernel::experimental::is_supported_matmul_operand_pair(IN0_DESC.shape, IN1_DESC.shape),
        "llk_math_matmul_init: unsupported in0/in1 tile pair (mismatched inner dimension, a 16-column in0 shorter "
        "than 32 rows, a short-face in0 with two face rows, an in1 with faces shorter than 16 rows, or an in1 other "
        "than 32x32 / 32x16 / 16x32)");
    SAN_HOOK(unsupported());
    constexpr std::uint32_t in0_tile_r_dim = IN0_DESC.shape.total_row_dim();
    constexpr std::uint32_t in0_tile_c_dim = IN0_DESC.shape.total_col_dim();
    constexpr std::uint32_t in1_tile_r_dim = IN1_DESC.shape.total_row_dim();
    constexpr std::uint32_t in1_tile_c_dim = IN1_DESC.shape.total_col_dim();
    // MATH-side rule == legacy (in0 tile_r < FACE_R_DIM). The UNPACK path uses the looser < TILE_R_DIM (see
    // llk_unpack_AB_matmul.h / llk_descriptor.h) -- the legacy MATH/UNPACK divergence, preserved.
    constexpr bool partial_face = in0_tile_r_dim < ckernel::FACE_R_DIM;
    // Compile-time form of the throttled program's runtime LLK_ASSERT. Where the execute path drives the fidelity
    // phases itself (throttle levels 4 / 5 at HiFi2 or higher) the init has no unthrottled fallback and always
    // builds the throttled program, which exists for the full 32x32-by-32x32 pair only; any other pair wedges
    // the math thread. Every input is a template argument here, so the pair is rejected at the call site.
    static_assert(
        !matmul_throttle_drives_phases<math_fidelity, THROTTLE_LEVEL>() ||
            matmul_throttle_supports_tile(in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face),
        "llk_math_matmul_init: MM_THROTTLE levels 4 and 5 at HiFi2 or higher keep the throttled matmul program, "
        "which supports the 32x32 by 32x32 operand pair only (any other pair wedges the math thread)");

    llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}
