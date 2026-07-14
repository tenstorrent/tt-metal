// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/mla_q_rope.hpp"

#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "metal/ops/mla_q_rope/mla_q_rope.hpp"

namespace ttml::ops {

autograd::TensorPtr mla_q_rope(
    const autograd::TensorPtr& q_full,
    const RotaryEmbeddingParams& rope_params,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    auto q_out = ttml::metal::mla_q_rope(
        q_full->get_value(),
        rope_params.cos_cache,
        rope_params.sin_cache,
        rope_params.trans_mat,
        qk_nope_dim,
        qk_rope_dim);

    auto out = autograd::create_tensor(q_out);

    // Backward rotates by -θ on the rope slice (neg cos/sin); nope passes through unchanged.
    autograd::GradFunction grad_fn = [q_full, rope_params, out, qk_nope_dim, qk_rope_dim]() {
        const auto& dL_dout = out->get_grad();

        const auto dL_dq_full = ttml::metal::mla_q_rope(
            dL_dout,
            rope_params.neg_cos_cache,
            rope_params.neg_sin_cache,
            rope_params.trans_mat,
            qk_nope_dim,
            qk_rope_dim);

        q_full->add_grad(dL_dq_full);
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, q_full));
    return out;
}

}  // namespace ttml::ops
