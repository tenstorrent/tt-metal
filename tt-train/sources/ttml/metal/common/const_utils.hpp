// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

}  // namespace ttml::metal
