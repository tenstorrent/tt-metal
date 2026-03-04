// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// SwiGLU forward operation: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
//
// Uses dual-NOC architecture with dynamic W2 prefetching for optimal performance.
// Requires M row (hidden_dim tiles) to fit in L1. For large hidden_dim, use
// sufficient tensor parallelism (TP) or fall back to composite ops.
//
// Args:
//   input_tensor: Input tensor [B, 1, S, embed_dim]
//   w1: Gate projection weights [1, 1, embed_dim, hidden_dim]
//   w2: Down projection weights [1, 1, hidden_dim, embed_dim]
//   w3: Up projection weights [1, 1, embed_dim, hidden_dim]
//
// Returns:
//   Output tensor [B, 1, S, embed_dim]
ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w2, const ttnn::Tensor& w3);

}  // namespace ttml::metal
