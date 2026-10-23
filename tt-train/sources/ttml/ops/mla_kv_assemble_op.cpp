// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_op.hpp"

#include <cstdint>
#include <tuple>
#include <utility>

#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/operations.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr> mla_kv_assemble(
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    auto [k_raw, v_raw] = ttml::metal::mla_kv_assemble_fw(
        kv_up->get_value(), k_pe->get_value(), n_heads, qk_nope_dim, qk_rope_dim, v_dim);

    auto out_k = autograd::create_tensor(k_raw);
    auto out_v = autograd::create_tensor(v_raw);

    autograd::GradFunction grad = [kv_up, k_pe, out_k, out_v, n_heads, qk_nope_dim, qk_rope_dim, v_dim]() {
        auto [dkv_up, dk_pe] = ttml::metal::mla_kv_assemble_bw(
            out_k->get_grad(), out_v->get_grad(), n_heads, qk_nope_dim, qk_rope_dim, v_dim);
        kv_up->add_grad(dkv_up);
        k_pe->add_grad(dk_pe);
    };

    auto k_node = autograd::add_backward_node(std::move(grad), out_k, kv_up, k_pe);
    if (k_node.has_value()) {
        out_k->set_node(k_node);
        // Sync node on out_v ensures dV is populated before grad lambda runs.
        out_v->set_node(autograd::add_backward_node_always([]() {}, out_v, kv_up, k_pe, out_k));
    }

    return {out_k, out_v};
}

}  // namespace ttml::ops
