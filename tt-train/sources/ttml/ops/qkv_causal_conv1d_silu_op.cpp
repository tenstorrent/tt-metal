// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_op.hpp"

#include <ttnn/operations/core/to_layout/to_layout_op.hpp>
#include <ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.hpp>
#include <unordered_map>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

namespace ttml::ops {

namespace {

// The forward's actual_start is a constant zero; one upload per device, so forwards stay free of host writes
// (which trace capture rejects).
ttnn::Tensor zero_actual_start(ttnn::distributed::MeshDevice* device) {
    static std::unordered_map<ttnn::distributed::MeshDevice*, ttnn::Tensor> cache;
    auto it = cache.find(device);
    if (it == cache.end() || !it->second.is_allocated() || it->second.device() != device) {
        it = cache
                 .insert_or_assign(
                     device,
                     core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                         std::vector<uint32_t>{0}, ttnn::Shape({1}), device, ttnn::Layout::ROW_MAJOR))
                 .first;
    }
    return it->second;
}

// ttml activations are [B,1,T,.]; the kernels take [1,T,.]. For TILE tensors these are views.
ttnn::Tensor rank3(const ttnn::Tensor& t) {
    const auto& shape = t.logical_shape();
    return shape.rank() == 4 ? ttnn::reshape(t, ttnn::Shape({1, shape[2], shape[3]})) : t;
}

ttnn::Tensor like_rank(const ttnn::Tensor& t, const ttnn::Shape& reference) {
    const auto& shape = t.logical_shape();
    return reference.rank() == 4 ? ttnn::reshape(t, ttnn::Shape({1, 1, shape[1], shape[2]})) : t;
}

}  // namespace

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> qkv_causal_conv1d_silu(
    const autograd::TensorPtr& input,
    const ttnn::Tensor& history,
    const autograd::TensorPtr& tap0,
    const autograd::TensorPtr& tap1,
    const autograd::TensorPtr& tap2,
    const autograd::TensorPtr& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size) {
    auto* device = &autograd::ctx().get_device();
    const auto input_shape = input->get_value().logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 || (input_shape.rank() == 4 && input_shape[0] == 1 && input_shape[1] == 1),
        "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V] or [1,1,T,Q+K+V], got {}",
        input_shape);
    auto x = rank3(input->get_value());
    // The backward gets the caller's value: 0 lets it pick its own chunk together with the row split.
    const uint32_t forward_chunk_size =
        channel_chunk_size != 0 ? channel_chunk_size
                                : metal::qkv_causal_conv1d_silu_chunk_size(q_width + k_width + v_width, *device);
    const ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig program_config{forward_chunk_size};
    auto history_rm = ttnn::to_layout(rank3(history), ttnn::Layout::ROW_MAJOR);
    auto actual_start = zero_actual_start(device);

    auto [q, k, v] = ttnn::experimental::kda::qkv_causal_conv1d_silu(
        x,
        history_rm,
        tap0->get_value(),
        tap1->get_value(),
        tap2->get_value(),
        tap3->get_value(),
        q_width,
        k_width,
        v_width,
        program_config,
        actual_start,
        history_rm);
    auto out_q = autograd::create_tensor(like_rank(q, input_shape));
    auto out_k = autograd::create_tensor(like_rank(k, input_shape));
    auto out_v = autograd::create_tensor(like_rank(v, input_shape));

    autograd::GradFunction grad = [input,
                                   x,
                                   tap0,
                                   tap1,
                                   tap2,
                                   tap3,
                                   out_q,
                                   out_k,
                                   out_v,
                                   history_rm,
                                   q_width,
                                   k_width,
                                   v_width,
                                   channel_chunk_size]() {
        // backward() seeds only its root: an output the loss does not use contributes zero.
        for (const auto& out : {out_q, out_k, out_v}) {
            if (!out->is_grad_initialized()) {
                out->set_grad(core::zeros_like(out->get_value()));
            }
        }
        // Only the backward reader wants rows; the forward reads the TILE input as is.
        auto input_rm = ttnn::to_layout(x, ttnn::Layout::ROW_MAJOR);
        auto [grad_input, grad_tap0, grad_tap1, grad_tap2, grad_tap3] = metal::qkv_causal_conv1d_silu_bw(
            rank3(out_q->get_grad()),
            rank3(out_k->get_grad()),
            rank3(out_v->get_grad()),
            input_rm,
            history_rm,
            tap0->get_value(),
            tap1->get_value(),
            tap2->get_value(),
            tap3->get_value(),
            q_width,
            k_width,
            v_width,
            channel_chunk_size,
            /*row_splits=*/0,
            /*compute_tap_grads=*/tap0->get_requires_grad() || tap1->get_requires_grad() || tap2->get_requires_grad() ||
                tap3->get_requires_grad());
        input->add_grad(like_rank(grad_input, input->get_value().logical_shape()));
        if (grad_tap0.has_value()) {
            tap0->add_grad(*grad_tap0);
            tap1->add_grad(*grad_tap1);
            tap2->add_grad(*grad_tap2);
            tap3->add_grad(*grad_tap3);
        }
    };
    auto q_node = autograd::add_backward_node(std::move(grad), out_q, input, tap0, tap1, tap2, tap3);
    out_q->set_node(q_node);
    if (q_node.has_value()) {
        out_k->set_node(autograd::add_backward_node_always([]() {}, out_k, input, out_q));
        out_v->set_node(autograd::add_backward_node_always([]() {}, out_v, input, out_q));
    }
    return {out_q, out_k, out_v};
}

}  // namespace ttml::ops
