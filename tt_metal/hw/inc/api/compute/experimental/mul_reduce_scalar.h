// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/eltwise_binary.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "experimental/llk_math_mul_reduce_scalar_api.h"
#endif
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_mul_reduce_scalar_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the fused multiply-reduce-scalar operation.
 *
 * This function initializes UNPACK and MATH for the fused
 * multiply + reduce scalar operation.
 *
 * Must be called before mul_reduce_scalar_tile().
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | Input circular buffer 0 (tensor A)                            | uint32_t | 0 to 31     | True     |
 * | icb1           | Input circular buffer 1 (tensor B)                            | uint32_t | 0 to 31     | True     |
 *
 * Return value: None
 */
// clang-format on
ALWI void mul_reduce_scalar_init(uint32_t icb0, uint32_t icb1) {
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));
    MATH((llk_math_eltwise_mul_reduce_scalar_init<MATH_FIDELITY>(icb0, false /*acc_to_dest*/)));
}

// clang-format off
/**
 * Performs a fused multiply-reduce-scalar operation on tiles.
 *
 * This function performs:
 * 1. Element-wise multiplication: C = A * B
 * 2. Scalar reduction: result = sum(all elements of C)
 *
 * The final scalar result is stored in dest[0] at element position [0].
 *
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | Input circular buffer 0 (tensor A)                            | uint32_t | 0 to 31     | True     |
 * | icb1           | Input circular buffer 1 (tensor B)                            | uint32_t | 0 to 31     | True     |
 * | num_tiles      | Number of tiles to process                                    | uint32_t | 1 to 8      | True     |
 * | scalar         | Scalar multiplier for reduction (default: 1.0)                | float    | Any float   | False    |
 *
 * Return value: None
 */
// clang-format on
template <PoolType reduce_type = PoolType::SUM>
ALWI void mul_reduce_scalar_tile(uint32_t icb0, uint32_t icb1, uint32_t num_tiles, float scaler = 1.0f) {
    // Step 1: Unpack input tiles from both circular buffers and perform multiplication
    for (uint32_t i = 0; i < num_tiles; i++) {
        UNPACK((llk_unpack_AB(icb0, icb1, i, i)));
        MATH((llk_math_eltwise_mul_reduce_scalar<DST_ACCUM_MODE, MATH_FIDELITY>(i, icb0)));
    }

    // Step 2: Switch UNPACK state for reduce phase (reset counters, set DVALID)
    UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));

    // Step 3: Initialize reduce operation
    MATH((llk_math_mul_reduce_scalar_reduce_init<DST_ACCUM_MODE, MATH_FIDELITY>()));

    // Step 4: Prepare data for first tile's scalar reduction
    // Move dest[0] (first multiply result) to srcA
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));

    // Populate srcB with the scaler value
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(
        _calculate_fill_, RC_custom, APPROX, 2 /*ITERATIONS*/, 0 /*dst_index*/, scaler));
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));

    // Clear dest[0] - this will accumulate scalar reduction results from all tiles
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(
        _calculate_fill_, RC_custom, APPROX, 2 /*ITERATIONS*/, 0 /*dst_index*/, 0.0f));

    // Step 5: Configure packer for scalar reduction
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, ReduceDim::REDUCE_SCALAR>()));

    // Step 6: Perform column reduction for each tile, accumulating into dest[0]
    // First iteration (i=0) - no move needed
    MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));

    // Remaining iterations - always move
    for (uint32_t i = 1; i < num_tiles; i++) {
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i)));
        MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));
    }

    // Step 7: Perform final scalar reduction
    MATH((llk_math_mul_reduce_scalar<MATH_FIDELITY>()));

    // Step 8: Clear data valid flags
    MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
}

// clang-format off
/**
 * Uninitializes the fused multiply-reduce-scalar operation.
 *
 * This function cleans up the reduce operation and should be called after
 * mul_reduce_scalar_tile() operations are complete.
 *
 * Return value: None
 */
// clang-format on
ALWI void mul_reduce_scalar_uninit() { PACK((llk_pack_reduce_mask_clear())); }

}  // namespace ckernel
