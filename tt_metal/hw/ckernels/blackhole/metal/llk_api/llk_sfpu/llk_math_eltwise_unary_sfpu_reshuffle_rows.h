// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_reshuffle_rows.h"

namespace ckernel {

/**
 * @brief Initialize SFPU for reshuffle rows operation
 *
 * Prepares the SFPU (Special Function Processing Unit) for gradient accumulation with row reshuffling.
 * This must be called once before using llk_math_eltwise_unary_sfpu_reshuffle_rows().
 *
 * Used in embedding backward pass to set up SFPU state for scatter-add operations where gradients
 * from different input positions are accumulated into their corresponding embedding rows.
 *
 */
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reshuffle_rows_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reshuffle_rows, APPROXIMATE>();
}

/**
 * @brief Perform gradient accumulation with row reshuffling using SFPU
 *
 * LLK API for embedding backward pass gradient accumulation. Implements the core
 * scatter-add operation: output[mask[i]] += input[i] for each row i in the input tile.
 *
 * This function processes a 32x32 tile where:
 * - Input tile (dst_index): Contains gradient data to be accumulated
 * - Mask data (idx_addr): Contains destination row mappings (uint8_t[32])
 * - Output: Accumulated gradients reshuffled according to mask (written to dst_index + 64)
 *
 * Algorithm:
 * - For each input row i (0-31): if mask[i] < 32, then output[mask[i]] += input[i]
 * - Mask value 255 means "skip this row" (no accumulation)
 * - Leverages SFPU vector operations for efficient parallel processing
 *
 * @note Must call llk_math_eltwise_unary_sfpu_reshuffle_rows_init() before first use
 * @note idx_addr must point to valid L1 memory accessible by TRISC1 compute thread
 */
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reshuffle_rows(
    uint dst_index, uint32_t idx_addr, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_reshuffle_rows<APPROXIMATE>, dst_index, vector_mode, idx_addr);
}

}  // namespace ckernel
