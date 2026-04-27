// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

// MoE FFN forward+backward with SwiGLU, for tokens already packed per local
// expert by the group op. For each local expert e:
//     Y_e = (SiLU(X_e @ W_gate_e) * (X_e @ W_up_e)) @ W_down_e
// where X_e = grouped[offsets[e] : offsets[e+1], :].
//
// `offsets` carries no gradient and stays a raw ttnn::Tensor.
//
// Shapes:
//   grouped : [1, 1, T_cap, H]   bf16 TILE DRAM
//   offsets : [E_local + 1]      uint32
//   w_gate  : [E_local, H, I]    bf16 TILE DRAM
//   w_up    : [E_local, H, I]    bf16 TILE DRAM
//   w_down  : [E_local, I, H]    bf16 TILE DRAM
// Returns:
//   Y       : [1, 1, T_cap, H]   bf16 TILE DRAM
autograd::TensorPtr moe_ffn_swiglu_fw(
    const autograd::TensorPtr& grouped,
    const ttnn::Tensor& offsets,
    const autograd::TensorPtr& w_gate,
    const autograd::TensorPtr& w_up,
    const autograd::TensorPtr& w_down);

}  // namespace ttml::ops
