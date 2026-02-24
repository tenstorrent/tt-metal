// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
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

#ifdef ARCH_BLACKHOLE
/**
 * Lightweight Blackhole-only reinit path used when reduce follows custom SDPA sub path.
 * Reprograms reduce MOP and restores only the reduce addrmods.
 */
template <uint32_t block_ct_dim>
ALWI void reduce_block_max_row_reinit_short() {
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE>()));
    MATH((llk_math_reduce_block_max_row_mop_config<block_ct_dim, DST_ACCUM_MODE>()));
    MATH((llk_math_reduce_block_max_row_reinit()));
}
#endif

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

}  // namespace ckernel
