// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ttml::serialization {

// ============================================================================
// Weight Loading Utilities
// ============================================================================
// These utilities are used for loading transformer weights from safetensors,
// handling format conversions required for RoPE (rotary position embedding) and
// dimension adjustments.

/**
 * @brief Pad and resize a flat weight vector to target dimensions
 *
 * Copies data from source to target, padding with small random values for any
 * additional rows/columns to avoid dead neurons.
 *
 * @param flat Input flat vector of weights
 * @param rows Source number of rows
 * @param cols Source number of columns
 * @param target_rows Target number of rows
 * @param target_cols Target number of columns
 * @return std::vector<float> Resized weight vector
 */
std::vector<float> pad_and_resize_flat(
    const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows, int64_t target_cols);

/**
 * @brief Unpermute projection weight rows for RoPE compatibility
 *
 * Reorders rows within each head from the non-meta format [0..D/2-1, D/2..D-1]
 * to interleaved format [0, D/2, 1, D/2+1, ...] expected by TTML RoPE.
 *
 * This is required for Q and K projection weights when loading from HuggingFace
 * format (non-meta style).
 *
 * @param w Input weight vector
 * @param rows Number of rows (num_heads * head_dim for Q, num_kv_heads * head_dim for K)
 * @param cols Number of columns (hidden_size)
 * @param n_heads Number of heads (num_heads for Q, num_kv_heads for K)
 * @return std::vector<float> Unpermuted weight vector
 */
std::vector<float> unpermute_proj_rows(const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads);

/**
 * @brief Unpermute RMSNorm weights for RoPE compatibility
 *
 * For RMSNorm weights (like Qwen3's q_norm/k_norm): reshape to (2, head_dim/2),
 * transpose, then flatten. This converts from non-meta format ([x1,x2,...,y1,y2...])
 * to the meta-style format expected by TTML ([x1,y1,x2,y2,...]).
 *
 * @param w Input weight vector
 * @return std::vector<float> Unpermuted weight vector
 */
std::vector<float> unpermute_norm_weights(const std::vector<float>& w);

/**
 * @brief Transpose a flat 2D weight matrix
 *
 * @param x Input flat vector
 * @param r Number of rows
 * @param c Number of columns
 * @return std::vector<float> Transposed flat vector
 */
std::vector<float> transpose_flat(const std::vector<float>& x, int64_t r, int64_t c);

/**
 * @brief Copy linear weights with optional transpose for shape matching
 *
 * Tries to match source weights to target dimensions, transposing if needed.
 * Throws if neither original nor transposed shape matches target.
 *
 * @param src Source weight vector
 * @param src_r Source rows
 * @param src_c Source columns
 * @param tgt_r Target rows
 * @param tgt_c Target columns
 * @param dbg Debug identifier for error messages
 * @param verbose If true, print when transposing
 * @return std::vector<float> Weights matching target dimensions
 */
std::vector<float> strict_copy_linear(
    const std::vector<float>& src,
    int64_t src_r,
    int64_t src_c,
    int64_t tgt_r,
    int64_t tgt_c,
    const std::string& dbg,
    bool verbose = false);

}  // namespace ttml::serialization
