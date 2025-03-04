// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rope_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace ttml::ops {

void RotaryEmbeddingParams::validate(const autograd::TensorPtr& input) const {
    if (input->get_rank() != 4) {
        throw std::runtime_error(
            "rope only supports rank-4 input tensors, but got rank " + std::to_string(input->get_rank()));
    }
    auto input_shape = input->get_shape();

    auto input_seq_len = input_shape[-2];
    auto input_head_dim = input_shape[-1];

    if (input_head_dim != head_dim) {
        throw std::runtime_error("RoPE input tensor's head dimension must match the head dimension in the params");
    }

    if (input_seq_len != sequence_length) {
        throw std::runtime_error("RoPE input tensor's sequence length must match the sequence length in the params");
    }

    auto trans_mat_shape = trans_mat.get_logical_shape();
    auto trig_param_shapes = std::array{
        cos_cache.get_logical_shape(),
        sin_cache.get_logical_shape(),
        neg_cos_cache.get_logical_shape(),
        neg_sin_cache.get_logical_shape()};

    if (!std::ranges::all_of(
            trig_param_shapes, [=](auto shape) { return shape == ttnn::Shape{1, 1, input_seq_len, input_head_dim}; })) {
        throw std::runtime_error(
            "All trigonometric rotary embedding parameters must have shape [1, 1, seq_len, head_dim]");
    }

    if (trans_mat_shape != ttnn::Shape{1, 1, 32, 32}) {
        throw std::runtime_error("RoPE trans mat must be of shape {1, 1, 32, 32}");
    }
}

// trans_mat, sin_cache, cos_cache all precomputed and stored somewhere in the module hierarchy
autograd::TensorPtr rope(const autograd::TensorPtr& input, const RotaryEmbeddingParams& params) {
    params.validate(input);

    // ensure everything in sight is interleaved over L1 before calling ttnn rope.
    auto to_l1 = [](const auto& t) { return ttnn::to_memory_config(t, ttnn::L1_MEMORY_CONFIG); };
    auto to_dram = [](const auto& t) { return ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG); };

    // FIXME: mostly use defaults for now, try tweaking.
    auto out_tensor = ttnn::experimental::rotary_embedding_llama(
        to_l1(input->get_value()), to_l1(params.cos_cache), to_l1(params.sin_cache), to_l1(params.trans_mat));
    auto out = autograd::create_tensor(to_dram(out_tensor));

    // In the backward pass we rotate by -θ, so we need negated cos and sin
    // caches. Note: we can just reuse trans_mat here since the data movement
    // should be the same on the backward pass (we use the same trick to speed
    // up the matmul, and the matrix used is specified by the cos/sin caches.)
    autograd::GradFunction grad_fn = [to_l1, to_dram, input, params, out]() {
        auto dL_dout = out->get_grad();
        auto dL_dinput = ttnn::experimental::rotary_embedding_llama(
            to_l1(dL_dout), to_l1(params.neg_cos_cache), to_l1(params.neg_sin_cache), to_l1(params.trans_mat));
        input->add_grad(to_dram(dL_dinput));
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad_fn), links));

    return out;
}

}  // namespace ttml::ops
