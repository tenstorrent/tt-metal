// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/eltwise_binary.h"

#ifdef TRISC_MATH
#include "llk_math_mul_reduce_scalar_api.h"
#include "sfpu/ckernel_sfpu_mul_reduce_scalar.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_mul_reduce_scalar_api.h"
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
    MATH((llk_math_mul_reduce_scalar_eltwise_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>(
        icb0, icb1, false /*acc_to_dest*/)));
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
 * | num_tiles      | Number of tiles to process                                    | uint32_t | 1+          | True     |
 *
 * Return value: None
 */
// clang-format on
template <PoolType reduce_type = PoolType::SUM>
ALWI void mul_reduce_scalar_tile(uint32_t icb0, uint32_t icb1, uint32_t num_tiles) {
    // Step 1: Unpack input tiles from both circular buffers and perform multiplication
    for (uint32_t i = 0; i < num_tiles; i++) {
        UNPACK((llk_unpack_AB(icb0, icb1, i, i)));
        MATH((llk_math_mul_reduce_scalar_eltwise<
              EltwiseBinaryType::ELWMUL,
              BroadcastType::NONE,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::NONE>(i)));
    }

    // Step 2: Switch UNPACK state for reduce phase (reset counters, set DVALID)
    UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));

    // Step 3: Initialize reduce operation
    MATH((llk_math_mul_reduce_scalar_reduce_init<reduce_type, ReduceDim::REDUCE_COL, DST_ACCUM_MODE, MATH_FIDELITY>()));

    // Step 4: Prepare data for first tile's scalar reduction
    // Move dest[0] (first multiply result) to srcA
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));

    // Populate srcB with ones for the scaler
    // Fill row 0 of Dest[0] with 1.0 (0x3F80 in bfloat16)
    MATH((ckernel::sfpu::populate_dest0_row_with_value_(0, 0x3F80)));
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));

    // Clear dest[0] - this will accumulate scalar reduction results from all tiles
    MATH((ckernel::sfpu::populate_dest0_row_with_value_(0, 0x0000)));

    // Step 5: Configure packer for scalar reduction
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, ReduceDim::REDUCE_SCALAR>()));

    // Step 6: Perform column reduction for each tile, accumulating into dest[0]
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (i != 0) {
            // Move dest[i] to srcA for next iteration
            MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i)));
        }
        // Perform column reduction, accumulating into dest[0]
        MATH((llk_math_mul_reduce_scalar_column<
              reduce_type,
              ReduceDim::REDUCE_COL,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              false,
              false,
              false>(0)));
    }

    // Step 7: Perform final scalar reduction
    MATH((llk_math_mul_reduce_scalar_final<
          reduce_type,
          ReduceDim::REDUCE_SCALAR,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          false,
          false>(0)));
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
