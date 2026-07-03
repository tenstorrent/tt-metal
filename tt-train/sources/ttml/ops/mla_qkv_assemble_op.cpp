// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_qkv_assemble_op.hpp"

#include <cstdint>
#include <tuple>
#include <utility>

#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/operations.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> mla_qkv_assemble(
    const autograd::TensorPtr& q_pre,
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    auto [q_raw, k_raw, v_raw] = ttml::metal::mla_qkv_assemble_fw(
        q_pre->get_value(), kv_up->get_value(), k_pe->get_value(), n_heads, qk_nope_dim, qk_rope_dim, v_dim);

    auto out_q = autograd::create_tensor(q_raw);
    auto out_k = autograd::create_tensor(k_raw);
    auto out_v = autograd::create_tensor(v_raw);

    autograd::GradFunction grad =
        [q_pre, kv_up, k_pe, out_q, out_k, out_v, n_heads, qk_nope_dim, qk_rope_dim, v_dim]() {
            auto [dq_pre, dkv_up, dk_pe] = ttml::metal::mla_qkv_assemble_bw(
                out_q->get_grad(), out_k->get_grad(), out_v->get_grad(), n_heads, qk_nope_dim, qk_rope_dim, v_dim);
            q_pre->add_grad(dq_pre);
            kv_up->add_grad(dkv_up);
            k_pe->add_grad(dk_pe);
        };

    auto q_node = autograd::add_backward_node(std::move(grad), out_q, q_pre, kv_up, k_pe);
    if (q_node.has_value()) {
        out_q->set_node(q_node);
        // Sync nodes on out_k and out_v ensure dK and dV are populated before grad lambda runs.
        out_k->set_node(autograd::add_backward_node_always([]() {}, out_k, q_pre, kv_up, k_pe, out_q));
        out_v->set_node(autograd::add_backward_node_always([]() {}, out_v, q_pre, kv_up, k_pe, out_q));
    }

    return {out_q, out_k, out_v};
}

}  // namespace ttml::ops
