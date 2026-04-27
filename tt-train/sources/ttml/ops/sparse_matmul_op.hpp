// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "autograd/tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

// Grouped (per-expert) matmul for tokens packed by the MoE group op.
//
// Shapes:
//   X       : [1, 1, T_cap, K]   bf16 TILE DRAM
//   W       : [E_local, K, N]    bf16 TILE DRAM
//   offsets : [E_local + 1]      uint32 (host)
// Returns:
//   Y       : [1, 1, T_cap, N]   bf16 TILE DRAM
//
// Composite implementation: ttnn::slice per expert + ttnn::concat. The
// natural zero-copy alternative via ttnn::narrow is not viable on the
// packed layout because narrow requires `dim_size % length == 0`, which
// per-expert offsets do not satisfy in general.

// Raw forward compute. Used by the autograd op and by callers that fuse
// multiple sparse matmuls into a single autograd node (e.g. MoE FFN).
ttnn::Tensor sparse_matmul_forward(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets);

// Raw backward compute. Returns (dX, dW). dW has shape [E_local, K, N];
// rows for empty experts are zero.
std::pair<ttnn::Tensor, ttnn::Tensor> sparse_matmul_backward(
    const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& dY, const ttnn::Tensor& offsets);

// Autograd-aware op. `offsets` carries no gradient (it's routing data) so
// it stays a raw ttnn::Tensor.
autograd::TensorPtr sparse_matmul(
    const autograd::TensorPtr& X, const autograd::TensorPtr& W, const ttnn::Tensor& offsets);

}  // namespace ttml::ops
