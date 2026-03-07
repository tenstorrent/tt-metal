// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// SwiGLU Gate-Up: M = SiLU(X @ W1) * (X @ W3)
//
// Uses 2D multicast matmul tiling for efficient DRAM bandwidth utilization.
// X is read once per K-block and shared across both W1 and W3 matmuls.
//
// Args:
//   input_tensor: Input tensor [B, 1, S, embed_dim]
//   w1: Gate projection weights [1, 1, embed_dim, hidden_dim]
//   w3: Up projection weights [1, 1, embed_dim, hidden_dim]
//
// Returns:
//   M tensor [B, 1, S, hidden_dim]
ttnn::Tensor swiglu_gate_up(const ttnn::Tensor& input_tensor, const ttnn::Tensor& w1, const ttnn::Tensor& w3);

}  // namespace ttml::metal
