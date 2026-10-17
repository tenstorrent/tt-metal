// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_op.hpp"

#include <ttnn/operations/core/to_layout/to_layout_op.hpp>
#include <ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

namespace ttml::ops {

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
    if (channel_chunk_size == 0) {
        channel_chunk_size = metal::qkv_causal_conv1d_silu_chunk_size(q_width + k_width + v_width, *device);
    }
    const ttnn::experimental::kda::QkvCausalConv1dSiluProgramConfig program_config{channel_chunk_size};
    auto input_rm = ttnn::to_layout(input->get_value(), ttnn::Layout::ROW_MAJOR);
    auto history_rm = ttnn::to_layout(history, ttnn::Layout::ROW_MAJOR);
    auto actual_start = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>{0}, ttnn::Shape({1}), device, ttnn::Layout::ROW_MAJOR);

    auto [q, k, v] = ttnn::experimental::kda::qkv_causal_conv1d_silu(
        input_rm,
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
    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad = [input,
                                   tap0,
                                   tap1,
                                   tap2,
                                   tap3,
                                   out_q,
                                   out_k,
                                   out_v,
                                   input_rm,
                                   history_rm,
                                   q_width,
                                   k_width,
                                   v_width,
                                   channel_chunk_size]() {
        auto [grad_input, grad_tap0, grad_tap1, grad_tap2, grad_tap3] = metal::qkv_causal_conv1d_silu_bw(
            out_q->get_grad(),
            out_k->get_grad(),
            out_v->get_grad(),
            input_rm,
            history_rm,
            tap0->get_value(),
            tap1->get_value(),
            tap2->get_value(),
            tap3->get_value(),
            q_width,
            k_width,
            v_width,
            channel_chunk_size);
        input->add_grad(grad_input);
        tap0->add_grad(grad_tap0);
        tap1->add_grad(grad_tap1);
        tap2->add_grad(grad_tap2);
        tap3->add_grad(grad_tap3);
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
