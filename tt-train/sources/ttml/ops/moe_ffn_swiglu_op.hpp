// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "autograd/tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

// MoE FFN forward+backward with SwiGLU, for tokens already packed per local
// expert by the group op. For each local expert e:
//     Y_e = (SiLU(X_e @ W_gate_e^T) * (X_e @ W_up_e^T)) @ W_down_e^T
// where X_e = grouped[offsets[e] : offsets[e+1], :].
//
// `offsets` carries no gradient and stays a raw ttnn::Tensor.
// Per-expert weights are passed as parallel lists of TensorPtr — one entry
// per local expert. Each weight tensor is consumed directly by `ttnn::matmul`
// (no slicing or stacking inside this op).
//
// Shapes:
//   grouped       : [1, 1, T_cap, hidden_dim]            bf16 TILE DRAM
//   offsets       : [E_local + 1]                        uint32
//   w_gate / w_up : E_local entries, each [1, 1, intermediate_dim, hidden_dim]
//   w_down        : E_local entries, each [1, 1, hidden_dim, intermediate_dim]
// (LinearLayer convention: weights stored as [out, in], matmul'd with transpose_b.)
// Returns:
//   Y             : [1, 1, T_cap, hidden_dim]            bf16 TILE DRAM
autograd::TensorPtr moe_ffn_swiglu_fw(
    const autograd::TensorPtr& grouped,
    const ttnn::Tensor& offsets,
    const std::vector<autograd::TensorPtr>& w_gate,
    const std::vector<autograd::TensorPtr>& w_up,
    const std::vector<autograd::TensorPtr>& w_down);

}  // namespace ttml::ops
