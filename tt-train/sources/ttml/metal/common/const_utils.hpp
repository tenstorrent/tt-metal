// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttml::metal {

/**
 * Specifies how attention masking is applied in scaled dot-product attention.
 *
 * - None: No mask applied, just scale attention scores
 * - Causal: Generate lower-triangular causal mask on device (prevents attending to future tokens)
 * - Arbitrary: Use provided mask tensor from DRAM
 */
enum class AttentionMaskType {
    None,      // No mask - just scale attention scores
    Causal,    // Generate causal mask on device (lower triangular)
    Arbitrary  // Use provided mask tensor from DRAM
};

/**
 * Specifies whether stochastic rounding is applied during optimizer updates.
 *
 * Stochastic rounding can help maintain training accuracy when using
 * reduced precision (e.g. BFLOAT16) by randomly rounding values up or down
 * based on their proximity to representable values.
 */
enum class StochasticRounding : bool { Disabled = false, Enabled = true };

/**
 * Specifies the output mode for symmetric gram matmul (G = X @ X^T).
 *
 * - UpperTriangle: Write only upper triangle + diagonal (lower triangle is uninitialized)
 * - Full: Write full symmetric matrix (upper + transposed mirror to lower triangle)
 */
enum class OutputMode : uint32_t {
    UpperTriangle = 0,
    Full = 1,
};

}  // namespace ttml::metal
