// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// sum(icb) * scaler, without an all-ones operand or an identity ELWMUL.
//
// Phase 1 copies each tile into DEST via datacopy (A2D). The reduce tail is the
// same DEST-only column-accumulate + one scalar collapse used by
// mul_reduce_scalar -- that protocol keeps SrcB (scaler) valid across tiles,
// which a mid-reduce unpack_A path does not.

#pragma once

#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/tile_move_copy.h"

namespace ckernel {

// Must be called before sum_reduce_scalar_tile().
ALWI void sum_reduce_scalar_init(uint32_t icb) { copy_tile_to_dst_init_short(icb); }

// Reduces num_tiles tiles of icb to a single scalar in dest[0] at [0, 0].
// Clobbers dest slots 0..num_tiles-1. Pair with mul_reduce_scalar_uninit().
ALWI void sum_reduce_scalar_tile(uint32_t icb, uint32_t num_tiles, float scaler = 1.0f) {
    for (uint32_t i = 0; i < num_tiles; i++) {
        copy_tile(icb, i, i);
    }

    UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));
    MATH((llk_math_mul_reduce_scalar_reduce_init<DST_ACCUM_MODE, MATH_FIDELITY>()));

    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(
        _calculate_fill_, RC_custom, APPROX, 2 /*ITERATIONS*/, 0 /*dst_index*/, scaler));
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(
        _calculate_fill_, RC_custom, APPROX, 2 /*ITERATIONS*/, 0 /*dst_index*/, 0.0f));

    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_SCALAR, PackMode::Default>()));

    MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb)));
    for (uint32_t i = 1; i < num_tiles; i++) {
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i)));
        MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb)));
    }

    MATH((llk_math_mul_reduce_scalar<MATH_FIDELITY>()));
    MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
}

}  // namespace ckernel
