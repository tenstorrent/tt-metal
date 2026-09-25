// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/eltwise_binary.h"
#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_fill.h"  // _calculate_fill_ used by mul_reduce_scalar_tile
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "experimental/llk_math_mul_reduce_scalar_api.h"
#endif
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_mul_reduce_scalar_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"
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
 * mul_reduce_scalar_init with an explicit fidelity for the multiply, for a caller whose multiply runs at a
 * fidelity other than the program's. MATH_FIDELITY exists only on the math thread, so it cannot be a default
 * template argument; this variant is named apart instead. Pair with mul_reduce_scalar_tile_fidelity.
 *
 * | Param Type | Name         | Description                              | Type         | Valid Range | Required |
 * |------------|--------------|------------------------------------------|--------------|-------------|----------|
 * | Template   | mul_fidelity | Fidelity of the element-wise multiply    | MathFidelity | N/A         | True     |
 * | Function   | icb0         | Input circular buffer 0 (tensor A)       | uint32_t     | 0 to 31     | True     |
 * | Function   | icb1         | Input circular buffer 1 (tensor B)       | uint32_t     | 0 to 31     | True     |
 */
// clang-format on
template <MathFidelity mul_fidelity>
ALWI void mul_reduce_scalar_init_fidelity(uint32_t icb0, uint32_t icb1) {
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));
    MATH((llk_math_eltwise_mul_reduce_scalar_init<mul_fidelity>(icb0, false /*acc_to_dest*/)));
}

namespace detail {
// Shared body of mul_reduce_scalar_tile and mul_reduce_scalar_tile_fidelity. program_fidelity selects
// MATH_FIDELITY for both phases, which is why it is only read inside MATH().
template <
    bool program_fidelity,
    MathFidelity mul_fidelity,
    MathFidelity reduce_fidelity,
    bool accumulate_in_one_tile,
    bool is_fp32_dest_acc_en>
ALWI void mul_reduce_scalar_tile_impl(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t num_tiles, float scaler) {
    MATH(constexpr MathFidelity mul_f = program_fidelity ? MATH_FIDELITY : mul_fidelity);
    MATH(constexpr MathFidelity reduce_f = program_fidelity ? MATH_FIDELITY : reduce_fidelity);

    // Step 1: Unpack input tiles from both circular buffers and perform multiplication. ELWMUL accumulates
    // into DEST, so with accumulate_in_one_tile every product lands in dest[0].
    for (uint32_t i = 0; i < num_tiles; i++) {
        UNPACK((llk_unpack_AB(icb0, icb1, i, i)));
        MATH((llk_math_eltwise_mul_reduce_scalar<is_fp32_dest_acc_en, mul_f>(accumulate_in_one_tile ? 0 : i, icb0)));
    }

    // Step 2: Switch UNPACK state for reduce phase (reset counters, set DVALID)
    UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));

    // Step 3: Initialize reduce operation
    MATH((llk_math_mul_reduce_scalar_reduce_init<is_fp32_dest_acc_en, reduce_f>()));

    // Step 4: Prepare data for first tile's scalar reduction
    // Move dest[0] (first multiply result) to srcA
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));

    // Populate srcB with the scaler value
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_fill_,
        (APPROX, 2 /*ITERATIONS*/),
        0 /*dst_index*/,
        VectorMode::RC_custom,
        scaler));
    MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));

    // Clear dest[0] - this will accumulate scalar reduction results from all tiles
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_fill_,
        (APPROX, 2 /*ITERATIONS*/),
        0 /*dst_index*/,
        VectorMode::RC_custom,
        0.0f));

    // Step 5: Configure packer for scalar reduction
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_SCALAR, PackMode::Default>(ocb)));

    // Step 6: Perform column reduction for each product tile, accumulating into dest[0]
    // First iteration (i=0) - no move needed
    MATH((llk_math_mul_reduce_column<reduce_f>(0, icb0)));

    // Remaining iterations - always move
    const uint32_t product_tiles = accumulate_in_one_tile ? 1 : num_tiles;
    for (uint32_t i = 1; i < product_tiles; i++) {
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i)));
        MATH((llk_math_mul_reduce_column<reduce_f>(0, icb0)));
    }

    // Step 7: Perform final scalar reduction
    MATH((llk_math_mul_reduce_scalar<reduce_f>()));

    // Step 8: Clear data valid flags
    MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
}
}  // namespace detail

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
 * The multiply accumulates into DEST, so the tiles it writes must be zero on entry; the packer clears each
 * DEST half it releases. By default product i goes to dest[i], which bounds num_tiles by the DEST capacity.
 * accumulate_in_one_tile sums every product in dest[0] instead: num_tiles is then unbounded, and one column
 * reduce serves the whole row. With a bf16 DEST each element's running sum is rounded to bf16 at every
 * accumulate, so its error grows with num_tiles.
 *
 * | Param Type | Name                   | Description                                                | Type     | Valid Range | Required |
 * |------------|------------------------|------------------------------------------------------------|----------|-------------|----------|
 * | Template   | accumulate_in_one_tile | Sum every product in dest[0] rather than one tile each     | bool     | true/false  | False    |
 * | Function   | icb0                   | Input circular buffer 0 (tensor A)                         | uint32_t | 0 to 31     | True     |
 * | Function   | icb1                   | Input circular buffer 1 (tensor B)                         | uint32_t | 0 to 31     | True     |
 * | Function   | ocb                    | Output circular buffer (used to program packer face_r_dim) | uint32_t | 0 to 31     | True     |
 * | Function   | num_tiles              | Number of tiles to process                                 | uint32_t | 1 to 8, or any with accumulate_in_one_tile | True |
 * | Function   | scalar                 | Scalar multiplier for reduction (default: 1.0)             | float    | Any float   | False    |
 *
 * Return value: None
 */
// clang-format on
template <
    PoolType reduce_type = PoolType::SUM,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    bool accumulate_in_one_tile = false>
ALWI void mul_reduce_scalar_tile(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t num_tiles, float scaler = 1.0f) {
    // The two fidelity arguments are placeholders: program_fidelity=true reads MATH_FIDELITY instead.
    detail::mul_reduce_scalar_tile_impl<
        true,
        MathFidelity::LoFi,
        MathFidelity::LoFi,
        accumulate_in_one_tile,
        is_fp32_dest_acc_en>(icb0, icb1, ocb, num_tiles, scaler);
}

// clang-format off
/**
 * mul_reduce_scalar_tile with an explicit fidelity for each phase, for a caller whose multiplies run at a
 * fidelity other than the program's. Pair with mul_reduce_scalar_init_fidelity<mul_fidelity>.
 *
 * | Param Type | Name                   | Description                                                | Type         | Valid Range | Required |
 * |------------|------------------------|------------------------------------------------------------|--------------|-------------|----------|
 * | Template   | mul_fidelity           | Fidelity of the element-wise multiply                      | MathFidelity | N/A         | True     |
 * | Template   | reduce_fidelity        | Fidelity of the column and scalar reduces                  | MathFidelity | N/A         | True     |
 * | Template   | accumulate_in_one_tile | As for mul_reduce_scalar_tile                              | bool         | true/false  | False    |
 *
 * The function arguments are those of mul_reduce_scalar_tile.
 *
 * Return value: None
 */
// clang-format on
template <
    MathFidelity mul_fidelity,
    MathFidelity reduce_fidelity,
    bool accumulate_in_one_tile = false,
    PoolType reduce_type = PoolType::SUM,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_reduce_scalar_tile_fidelity(
    uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t num_tiles, float scaler = 1.0f) {
    detail::
        mul_reduce_scalar_tile_impl<false, mul_fidelity, reduce_fidelity, accumulate_in_one_tile, is_fp32_dest_acc_en>(
            icb0, icb1, ocb, num_tiles, scaler);
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
