// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/eltwise_binary.h"
#if defined(ARCH_BLACKHOLE)
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#endif
#include "llk_assert.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_fill.h"  // _calculate_fill_ used by mul_reduce_scalar_tile
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
    // The reduce tail's move_dest_to_src only encodes tiles 0..7, so 8 caps num_tiles even when DEST holds more.
    constexpr uint32_t dest_capacity =
        get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
    constexpr uint32_t max_tiles = dest_capacity < 8 ? dest_capacity : 8;
    LLK_ASSERT(
        accumulate_in_one_tile || num_tiles <= max_tiles,
        "mul_reduce_scalar_tile: num_tiles exceeds min(8, DEST capacity); accumulate_in_one_tile has no such limit");
#if !defined(ARCH_BLACKHOLE)
    static_assert(!accumulate_in_one_tile, "accumulate_in_one_tile is Blackhole-only");
#endif

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
 * DEST half it releases. By default product i goes to dest[i], which bounds num_tiles by min(8, DEST capacity).
 * accumulate_in_one_tile sums every product in dest[0] instead: num_tiles is then unbounded, and one column
 * reduce serves the whole row. With a bf16 DEST each element's running sum is rounded to bf16 at every
 * accumulate, so its error grows with num_tiles.
 *
 * | Param Type | Name                   | Description                                                | Type     | Valid Range | Required |
 * |------------|------------------------|------------------------------------------------------------|----------|-------------|----------|
 * | Template   | accumulate_in_one_tile | Sum every product in dest[0] rather than one tile each (Blackhole only) | bool | true/false | False |
 * | Function   | icb0                   | Input circular buffer 0 (tensor A)                         | uint32_t | 0 to 31     | True     |
 * | Function   | icb1                   | Input circular buffer 1 (tensor B)                         | uint32_t | 0 to 31     | True     |
 * | Function   | ocb                    | Output circular buffer (used to program packer face_r_dim) | uint32_t | 0 to 31     | True     |
 * | Function   | num_tiles              | Number of tiles to process                                 | uint32_t | 1 to min(8, DEST tile capacity) (4 in fp32 half-sync), or any with accumulate_in_one_tile | True |
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

#if defined(ARCH_BLACKHOLE)

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

/**
 * Reduce a row of products whose tile count exceeds DST capacity.
 *
 * One DST slot is reserved as a cross-chunk accumulator; the remaining
 * dst_capacity - 1 slots stage products. The caller must initialize
 * mul_reduce_scalar and add_binary, then acquire DST before calling.
 *
 * ocb programs the packer's face_r_dim for the reduce mask. On return,
 * DST[dst_capacity - 1] contains scaler * scaler * sum(A * B): the column
 * and row reductions both apply scaler, so a mean requires 1 / sqrt(width).
 * The reduce pack mask remains configured; call
 * mul_reduce_scalar_uninit() before normal packing.
 */
template <
    uint32_t num_tiles,
    uint32_t dst_capacity,
    PoolType reduce_type = PoolType::SUM,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_reduce_scalar_chunked_tile(uint32_t icb0, uint32_t icb1, uint32_t ocb, float scaler = 1.0f) {
    static_assert(reduce_type == PoolType::SUM, "Only SUM reduction is currently supported");
    static_assert(dst_capacity >= 2 && dst_capacity <= 8, "Chunked reduction requires 2 to 8 DST slots");
    static_assert(
        dst_capacity <= get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>(),
        "dst_capacity exceeds the DST tiles available in this sync/accum mode");
    static_assert(num_tiles > dst_capacity, "Use mul_reduce_scalar_tile when the row fits DST");

    constexpr uint32_t batch_size = dst_capacity - 1;
    constexpr uint32_t accumulator = batch_size;
    constexpr uint32_t num_batches = (num_tiles + batch_size - 1) / batch_size;
    constexpr uint32_t last_batch_size = num_tiles - (num_batches - 1) * batch_size;

    fill_tile(accumulator, 0.0f);

    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        const uint32_t input_start = batch * batch_size;
        const uint32_t count = batch + 1 < num_batches ? batch_size : last_batch_size;

        // Each reduction consumes the UNPACK/MATH state. The caller provides
        // the first initialization; subsequent chunks restore it here.
        if (batch > 0) {
            mul_reduce_scalar_init(icb0, icb1);
        }
        for (uint32_t j = 0; j < count; ++j) {
            // Products reuse DEST slots across chunks; preserve the running
            // accumulator while clearing only the next multiplication's tile.
            MATH((llk_math_mul_reduce_scalar_clear_product_tile<dst_capacity, is_fp32_dest_acc_en>(j)));
            UNPACK((llk_unpack_AB(icb0, icb1, input_start + j, input_start + j)));
            MATH((llk_math_eltwise_mul_reduce_scalar<is_fp32_dest_acc_en, MATH_FIDELITY>(j, icb0)));
        }

        UNPACK((llk_unpack_mul_reduce_scalar_switch_to_reduce()));
        MATH((llk_math_mul_reduce_scalar_reduce_init<is_fp32_dest_acc_en, MATH_FIDELITY>()));
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0)));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_fill_,
            (APPROX, 2 /*ITERATIONS*/),
            0 /*dst_index*/,
            VectorMode::RC_custom,
            scaler));
        MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0)));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_fill_,
            (APPROX, 2 /*ITERATIONS*/),
            0 /*dst_index*/,
            VectorMode::RC_custom,
            0.0f));

        if (batch == 0) {
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_SCALAR, ckernel::PackMode::Default>(ocb)));
        }
        MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));
        for (uint32_t j = 1; j < count; ++j) {
            MATH((llk_math_mul_reduce_scalar_move_dest_to_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(j)));
            MATH((llk_math_mul_reduce_column<MATH_FIDELITY>(0, icb0)));
        }
        MATH((llk_math_mul_reduce_scalar<MATH_FIDELITY>()));
        MATH((llk_math_mul_reduce_scalar_clear_dvalid()));
        add_binary_tile<DstRoundingMode::Default, is_fp32_dest_acc_en>(accumulator, 0, accumulator);
    }
}

#endif  // ARCH_BLACKHOLE

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
