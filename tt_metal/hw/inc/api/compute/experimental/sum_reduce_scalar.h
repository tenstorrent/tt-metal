// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/tile_move_copy.h"

namespace ckernel {

// clang-format off
/**
 * Initializes the sum-reduce-scalar operation.
 *
 * Configures UNPACK and MATH for the datacopy phase. The reduce phase reconfigures
 * both of those threads itself, so only the datacopy needs setting up here.
 *
 * Must be called before sum_reduce_scalar_tile().
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb            | Input circular buffer                                         | uint32_t | 0 to 31     | True     |
 *
 * Return value: None
 */
// clang-format on
ALWI void sum_reduce_scalar_init(uint32_t icb) { copy_init(icb); }

// clang-format off
/**
 * Reduces num_tiles tiles of icb to a single scalar:
 *
 *     result = sum(all elements of all tiles) * scaler^2
 *
 * NOTE the square: scaler is loaded into SrcB once and both GAPOOL passes (the
 * per-tile column accumulate and the final scalar collapse) multiply by it, so it
 * lands on the result twice. To scale the sum by 1/N, pass 1/sqrt(N).
 *
 * The result is stored in dest[0] at element position [0]; every other lane is
 * unspecified. Slots 0..num_tiles-1 of DEST are clobbered.
 *
 * num_tiles is bounded twice. The hard ceiling is 8: the shared reduce tail moves
 * dest[i] to SrcA through a switch that only covers i in 0..7 and no-ops above that,
 * so a 9th tile would be silently dropped from the sum. Below that, the acquired DEST
 * must also hold every copied tile until the reduce consumes it, which caps num_tiles
 * at get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>() --
 * 8 for half-sync/16-bit, 4 for half-sync/32-bit, 8 for full-sync/32-bit. Callers
 * pairing fp32 DEST with half-sync therefore get 4, not 8.
 *
 * This is the sum-only counterpart to mul_reduce_scalar_tile(): it reaches the same
 * reduction without an all-ones second operand or an identity ELWMUL. Phase 1 copies
 * each tile into DEST via datacopy (A2D); the reduce tail is the same DEST-only
 * column-accumulate plus scalar collapse, which is what keeps SrcB (the scaler) valid
 * across tiles — a mid-reduce unpack_A path does not.
 *
 * Pair with mul_reduce_scalar_uninit().
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb            | Input circular buffer                                         | uint32_t | 0 to 31     | True     |
 * | ocb            | Output circular buffer (used to program packer face_r_dim)    | uint32_t | 0 to 31     | True     |
 * | num_tiles      | Number of tiles to reduce (see the DEST note above)           | uint32_t | 1 to 8      | True     |
 * | scaler         | Per-GAPOOL multiplier; applied twice (default: 1.0)           | float    | Any float   | False    |
 *
 * Return value: None
 */
// clang-format on
ALWI void sum_reduce_scalar_tile(uint32_t icb, uint32_t ocb, uint32_t num_tiles, float scaler = 1.0f) {
    // Step 1: Copy each input tile into its own DEST slot
    for (uint32_t i = 0; i < num_tiles; i++) {
        copy_tile(icb, i, i);
    }

    // Step 2: Switch UNPACK state for reduce phase (reset counters, set DVALID)
    UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));

    // Step 3: Initialize reduce operation
    MATH((llk_math_mul_reduce_scalar_reduce_init<DST_ACCUM_MODE, MATH_FIDELITY>()));

    // Step 4: Move dest[0] (first copied tile) to srcA
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));

    // Populate srcB with the scaler value
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, _calculate_fill_, (APPROX, 2 /*ITERATIONS*/), 0 /*dst_index*/, VectorMode::RC_custom, scaler));
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));

    // Clear dest[0] - this will accumulate scalar reduction results from all tiles
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, _calculate_fill_, (APPROX, 2 /*ITERATIONS*/), 0 /*dst_index*/, VectorMode::RC_custom, 0.0f));

    // Step 5: Configure packer for scalar reduction
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_SCALAR, PackMode::Default>(ocb)));

    // Step 6: Column-reduce each tile, accumulating into dest[0]
    MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb)));
    for (uint32_t i = 1; i < num_tiles; i++) {
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i)));
        MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb)));
    }

    // Step 7: Perform final scalar reduction
    MATH((llk_math_mul_reduce_scalar<MATH_FIDELITY>()));

    // Step 8: Clear data valid flags
    MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
}

}  // namespace ckernel
