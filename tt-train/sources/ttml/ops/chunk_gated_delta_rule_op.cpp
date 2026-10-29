// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_op.hpp"

#include <optional>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "metal/operations.hpp"

namespace ttml::ops {

namespace {

ttnn::Tensor as_dtype(const ttnn::Tensor& tensor, tt::tt_metal::DataType dtype) {
    return tensor.dtype() == dtype ? tensor : ttnn::typecast(tensor, dtype);
}

}  // namespace

autograd::TensorPtr chunk_gated_delta_rule(
    const autograd::TensorPtr& q,
    const autograd::TensorPtr& k,
    const autograd::TensorPtr& v,
    const autograd::TensorPtr& g,
    const autograd::TensorPtr& beta,
    uint32_t key_head_dim,
    uint32_t chunk_size,
    std::optional<float> scale,
    const autograd::TensorPtr& initial_state) {
    const auto dtype = q->get_value().dtype();

    // The forward takes the state in FLOAT32 and the backward in the inputs' dtype.
    std::optional<ttnn::Tensor> h0_fw;
    std::optional<ttnn::Tensor> h0_bw;
    if (initial_state != nullptr) {
        h0_fw = as_dtype(initial_state->get_value(), tt::tt_metal::DataType::FLOAT32);
        h0_bw = as_dtype(initial_state->get_value(), dtype);
    }

    // The saved state only feeds the backward; skip writing it when no backward node will exist.
    const bool save_for_backward = autograd::ctx().get_gradient_mode() != autograd::GradMode::DISABLED &&
                                   autograd::any_requires_grad(q, k, v, g, beta, initial_state);
    auto fw = ttml::metal::chunk_gated_delta_rule_fw(
        q->get_value(),
        k->get_value(),
        v->get_value(),
        g->get_value(),
        beta->get_value(),
        key_head_dim,
        h0_fw,
        chunk_size,
        scale,
        /*output_head_major=*/true,
        /*compute_kernel_config=*/std::nullopt,
        save_for_backward);
    auto out = autograd::create_tensor(as_dtype(fw.output, dtype));

    autograd::GradFunction grad = [q,
                                   k,
                                   v,
                                   g,
                                   beta,
                                   initial_state,
                                   out,
                                   h0_bw,
                                   h = fw.h,
                                   v_new = fw.v_new,
                                   a = fw.a,
                                   key_head_dim,
                                   chunk_size,
                                   scale,
                                   dtype]() {
        TT_FATAL(
            h.has_value() && v_new.has_value() && a.has_value(), "chunk_gated_delta_rule: the forward saved no state");
        // Leaves that do not require a gradient get none: skip the reduction and the casts that follow.
        const ttml::metal::ChunkGatedDeltaRuleBwNeeds needs{
            .dq = q->get_requires_grad(),
            .dk = k->get_requires_grad(),
            .dv = v->get_requires_grad(),
            .dg = g->get_requires_grad(),
            .dbeta = beta->get_requires_grad(),
            .dh0 = initial_state != nullptr && initial_state->get_requires_grad(),
        };
        auto grads = ttml::metal::chunk_gated_delta_rule_bw(
            q->get_value(),
            k->get_value(),
            v->get_value(),
            g->get_value(),
            beta->get_value(),
            as_dtype(out->get_grad(), dtype),
            key_head_dim,
            h0_bw,
            /*d_final_state=*/std::nullopt,
            h,
            v_new,
            a,
            chunk_size,
            scale,
            /*compute_kernel_config=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            needs);

        if (grads[0]) {
            q->add_grad(*grads[0]);
        }
        if (grads[1]) {
            k->add_grad(*grads[1]);
        }
        if (grads[2]) {
            v->add_grad(*grads[2]);
        }
        // dg / dbeta come back in q's dtype; the gates may be in the other one.
        if (grads[3]) {
            g->add_grad(as_dtype(*grads[3], g->get_value().dtype()));
        }
        if (grads[4]) {
            beta->add_grad(as_dtype(*grads[4], beta->get_value().dtype()));
        }
        if (grads[5]) {
            initial_state->add_grad(as_dtype(*grads[5], initial_state->get_value().dtype()));
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, q, k, v, g, beta, initial_state));
    return out;
}

}  // namespace ttml::ops
