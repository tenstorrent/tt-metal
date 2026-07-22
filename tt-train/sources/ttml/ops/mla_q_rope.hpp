// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "ops/rope_op.hpp"

namespace ttml::ops {

// Fused MLA Q RoPE forward and backward (metal mla_q_rope; backward uses neg cos/sin on the rope slice).
autograd::TensorPtr mla_q_rope(
    const autograd::TensorPtr& q_full,
    const RotaryEmbeddingParams& rope_params,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim);

}  // namespace ttml::ops
