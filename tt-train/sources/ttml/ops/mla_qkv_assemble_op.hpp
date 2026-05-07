// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Autograd-aware fused MLA QKV assembly. See ``ttml::metal::mla_qkv_assemble_fw``
// for shape semantics. Forward dispatches to the metal kernel; backward is
// composed from existing ttnn primitives (sum, slice, concat, transpose,
// reshape) and registered with the C++ autograd graph.
//
// Returns {q, k_full, v}. Q is head-split but NOT yet RoPE'd — the caller
// applies ``rope_partial`` afterward.
std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> mla_qkv_assemble(
    const autograd::TensorPtr& q_pre,
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttml::ops
