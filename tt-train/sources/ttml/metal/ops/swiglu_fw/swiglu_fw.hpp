// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// SwiGLU forward operation: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
//
// Paths:
//   GateUp:     Fused gate-up kernel + matmul(M, W2). Best when M fits in L1.
//   FullFusion: Single fused kernel (legacy).
//   Composite:  Same op sequence as LlamaMLP: matmul, silu, matmul, mul, matmul.
//               For perf comparison with llama_block composite.
//
// Args:
//   input_tensor: Input tensor [B, 1, S, embed_dim]
//   w1: Gate projection weights [1, 1, embed_dim, hidden_dim]
//   w2: Down projection weights [1, 1, hidden_dim, embed_dim]
//   w3: Up projection weights [1, 1, embed_dim, hidden_dim]
//   path: Which implementation to use (default: GateUp).
//
// Returns:
//   Output tensor [B, 1, S, embed_dim]
enum class SwigluFwPath { Composite, GateUp, FullFusion };

ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    const ttnn::Tensor& w3,
    SwigluFwPath path = SwigluFwPath::GateUp);

}  // namespace ttml::metal
