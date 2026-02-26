// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce_custom.h"

/*************************************************************************
 * LLK REDUCE CUSTOM - Specialized reduce_max_row operations
 *************************************************************************/

// Block-based reduce row max functions
/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native llk_math_reduce_init LLK.
 * Use the standard llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * llk_math_reduce() calls in a loop for general-purpose block reduction.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init() {
    _llk_math_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>();
}

template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_mop_config() {
    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>();
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native llk_math_reduce LLK.
 * Use the standard llk_math_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row(const uint dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    _llk_math_reduce_block_max_row_<block_ct_dim, is_fp32_dest_acc_en>(dst_index);
}

/**
 * Reinitializes the block-based reduce_max_row operation after a matmul.
 *
 * This LLK API function is used only to re-initialize the address modifiers after a
 * matmul operation in an SDPA inner loop. Please don't use this function as a substitute for
 * the native llk_math_reduce_block_max_row_init LLK. This function is highly specialized
 * for a certain use case and the LLK team does not guarantee any degree of generality.
 */
inline void llk_math_reduce_block_max_row_reinit() { reduce_max_row_configure_addrmod_reinit(); }
