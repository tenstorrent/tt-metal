// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

// MoE FFN forward with SwiGLU, for tokens already packed per local expert by
// the group op. For each local expert e the slice
// grouped[offsets[e] : offsets[e+1], :] is passed through
//     Y_e = (SiLU(X_e @ W_gate_e) * (X_e @ W_up_e)) @ W_down_e.
// Row layout of the output matches `grouped` so the same offsets/plan/token_map
// drive the downstream combine.
//
// Shapes:
//   grouped : [1, 1, T_cap, H]      bf16 TILE DRAM
//   offsets : [E_local + 1]         uint32 L1
//   w_gate  : [E_local, H, I]       bf16 TILE DRAM
//   w_up    : [E_local, H, I]       bf16 TILE DRAM
//   w_down  : [E_local, I, H]       bf16 TILE DRAM
// Returns:
//   Y       : [1, 1, T_cap, H]      bf16 TILE DRAM
ttnn::Tensor moe_ffn_swiglu_fw(
    const ttnn::Tensor& grouped,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down);

}  // namespace ttml::ops
