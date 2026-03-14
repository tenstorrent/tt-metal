// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Full-fusion SwiGLU forward: Y = (SiLU(X @ W1) * (X @ W3)) @ W2 in a single fused kernel.
// Path selection (Composite, GateUp, FullFusion) lives in ttml::ops::swiglu (swiglu_op.cpp).
//
// Args:
//   input_tensor: [B, 1, S, embed_dim]
//   w1, w2, w3: Gate, down, up projections [1, 1, embed_dim, hidden_dim] etc.
// Returns:
//   Output tensor [B, 1, S, embed_dim]
ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w2, const ttnn::Tensor& w3);

}  // namespace ttml::metal
