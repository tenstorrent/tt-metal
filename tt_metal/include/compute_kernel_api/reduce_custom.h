// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"
#include "llk_math_reduce_custom_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_AB_reduce_custom_api.h"
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initialization for reduce_block_max_row operation. Must be called before reduce_block_max_row.
 * Processes a block of tiles in the width dimension, reducing each row across the block.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on DST_ACCUM_MODE
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_init API.
 * Use the standard reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with reduce_tile() in a loop
 * for general-purpose reduction across multiple tiles.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | block_ct_dim              | The number of tiles in the width dimension to process as a block                        | uint32_t  | 1 to 2^32-1                                   | True     |
 */
// clang-format on
template <uint32_t block_ct_dim>
ALWI void reduce_block_max_row_init() {
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE>()));
    MATH((llk_math_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE>()));
    PACK((llk_pack_reduce_mask_config<false, ReduceDim::REDUCE_ROW>()));
}

// clang-format off
/**
 * Performs block-based max row reduction operation on a block of tiles in the width dimension.
 * Reduces each row across the block of tiles and writes the result to DST register.
 * The DST register buffer must be in acquired state via acquire_dst call.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on DST_ACCUM_MODE
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_tile API.
 * Use the standard reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with reduce_tile() in a loop
 * for general-purpose reduction across multiple tiles.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | block_ct_dim              | The number of tiles in the width dimension to process as a block                        | uint32_t  | 1 to 2^32-1                                   | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler                | CB holding scaling factors                                                              | uint32_t  | 0 to 31                                        | True     |
 * | Function   | row_start_index           | The starting tile index for the row being processed                                     | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | idst                      | The index of the tile in DST REG for the result                                         | uint32_t  | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <uint32_t block_ct_dim>
ALWI void reduce_block_max_row(uint32_t icb, uint32_t icb_scaler, uint32_t row_start_index, uint32_t idst) {
    UNPACK((llk_unpack_AB_reduce_block_max_row<block_ct_dim>(icb, icb_scaler, row_start_index)));
    MATH((llk_math_reduce_block_max_row<block_ct_dim, DST_ACCUM_MODE>(idst)));
}

// clang-format off
/**
 * Uninitializes the block-based reduce_max_row operation. Needs to be called after the last call to `reduce_block_max_row` before initializing another operation.
 * This version is for block-based reduction across multiple tiles processed together.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on clear_fp32_accumulation flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_uninit API.
 * Use the standard reduce_uninit() for general-purpose reduction cleanup.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | clear_fp32_accumulation   | Whether to clear FP32 accumulation state                                                | bool      | {true, false}                                  | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A. Required when clear_fp32_accumulation=true | uint32_t  | 0 to 31 | Conditional |
 */
// clang-format on
template <bool clear_fp32_accumulation = false>
ALWI void reduce_block_max_row_uninit(uint32_t icb) {
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_reduce_uninit<clear_fp32_accumulation>()));
#else
    // Required because MOVB2D/D2B depends on SrcA ALU Format - Hi/Lo16 does not work with Tf32 (only on WH)
    // This is needed because FP32 data from L1 that is unpacked to Src registers is reduced to Tf32
    // See _llk_math_reduce_init_ for more details
    MATH((llk_math_reduce_uninit<clear_fp32_accumulation>(icb)));
#endif
    PACK((llk_pack_reduce_mask_clear()));
    UNPACK((llk_unpack_AB_reduce_block_max_row_uninit()));
}

// clang-format off
/**
 * WORK IN PROGRESS - Use with caution
 *
 * L1 → DEST: Block-level reduce operation.
 * For-loop wrapper around reduce_tile(). Use reduce_init() before calling.
 * Result stays in DEST for SFPU fusion or further operations.
 * Conforms to Compute API Contract for *_block variants.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type               | The type of reduce op - sum, average or maximum                                         | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim                | The dimension of reduce op - row, column or both                                        | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Template   | Ht                        | Block height in tiles (compile-time)                                                    | uint32_t  | 1 to 16                                        | True     |
 * | Template   | Wt                        | Block width in tiles (compile-time)                                                     | uint32_t  | 1 to 16                                        | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler                | CB holding scaling factors                                                              | uint32_t  | 0 to 31                                        | True     |
 * | Function   | itile_start               | Starting tile index in CB                                                               | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | itile_scaler              | The index of the tile within the scaling factor CB                                      | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | idst_start                | Starting tile index in DST REG for the result                                           | uint32_t  | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, uint32_t Ht, uint32_t Wt>
ALWI void reduce_block(
    uint32_t icb, uint32_t icb_scaler, uint32_t itile_start, uint32_t itile_scaler, uint32_t idst_start) {
    static_assert(
        Ht * Wt <= 16, "Block size Ht * Wt exceeds DEST capacity (max 16 tiles)");

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t w = 0; w < Wt; w++) {
            uint32_t tile_offset = h * Wt + w;
            reduce_tile<reduce_type, reduce_dim>(
                icb, icb_scaler, itile_start + tile_offset, itile_scaler, idst_start + tile_offset);
        }
    }
}

}  // namespace ckernel
