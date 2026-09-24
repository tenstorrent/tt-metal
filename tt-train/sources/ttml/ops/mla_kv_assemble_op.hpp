// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Autograd-aware fused MLA KV assembly. Forward and backward each dispatch to a dedicated Metal op
// (``ttml::metal::mla_kv_assemble_fw`` / ``mla_kv_assemble_bw``). See those for shape semantics.
// Q head-split + RoPE is handled separately by ``mla_q_rope``.
//
// Returns {k, v}.
std::tuple<autograd::TensorPtr, autograd::TensorPtr> mla_kv_assemble(
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttml::ops
