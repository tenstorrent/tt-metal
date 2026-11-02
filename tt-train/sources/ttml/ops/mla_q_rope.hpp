// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ops/rope_op.hpp"

namespace ttml::ops {

// Fused MLA Q RoPE + head-split. Forward: packed q_pre -> head-major q_roped.
// Backward: head-major dL/dq_roped -> packed dq_pre (neg cos/sin on the rope slice).
//
// q_pre: [B, 1, S, n_heads * (qk_nope_dim + qk_rope_dim)]
// out:   [B, n_heads, S, qk_nope_dim + qk_rope_dim]
autograd::TensorPtr mla_q_rope(
    const autograd::TensorPtr& q_pre,
    const RotaryEmbeddingParams& rope_params,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim);

}  // namespace ttml::ops
