// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Autograd-aware fused MLA QKV assembly. Forward and backward each dispatch to a dedicated Metal op
// (``ttml::metal::mla_qkv_assemble_fw`` / ``mla_qkv_assemble_bw``). See those for shape semantics.
//
// Returns {q, k, v}. Q is head-split but NOT yet RoPE'd; the caller applies ``rope_trailing`` afterward.
std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> mla_qkv_assemble(
    const autograd::TensorPtr& q_pre,
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttml::ops
