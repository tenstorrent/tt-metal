// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/swiglu_fw_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Re-export algorithm enum for convenience
using SwiGLUAlgorithm = ops::swiglu_fw::device::SwiGLUAlgorithm;

// SwiGLU forward operation: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
//
// Args:
//   input_tensor: Input tensor [B, 1, S, embed_dim]
//   w1: Gate projection weights [1, 1, embed_dim, hidden_dim]
//   w2: Down projection weights [1, 1, hidden_dim, embed_dim]
//   w3: Up projection weights [1, 1, embed_dim, hidden_dim]
//   algorithm: Algorithm selection (default: AUTO)
//     - AUTO: Automatically select based on L1 availability
//     - ORIGINAL: Materialize full M row in L1 (faster for small hidden_dim)
//     - TRUE_FLASH: Compute M on-demand (50% less L1, prepares for block matmul)
//
// Returns:
//   Output tensor [B, 1, S, embed_dim]
ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    const ttnn::Tensor& w3,
    SwiGLUAlgorithm algorithm = SwiGLUAlgorithm::AUTO);

}  // namespace ttml::metal
