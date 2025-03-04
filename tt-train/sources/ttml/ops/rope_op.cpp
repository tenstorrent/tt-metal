// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rope_op.hpp"

#include <xtensor/xmanipulation.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "core/xtensor_utils.hpp"
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

std::pair<ttnn::Tensor, ttnn::Tensor> gen_freqs(uint32_t head_dim, uint32_t sequence_length, float theta = 10000.0F) {
    int d = head_dim;
    // compute freqs: 1.0 / (theta ** (2 * (i-1) / head_dim)) for i in [1, head_dim/2]
    std::vector<float> expt_data;
    expt_data.reserve(d);
    for (uint32_t i = 1; i <= d / 2; i++) {
        expt_data.push_back(static_cast<float>(i - 1));
        expt_data.push_back(static_cast<float>(i - 1));
    }
    xt::xarray<float> expt_xt = xt::adapt(expt_data);
    expt_xt *= 2.0F / static_cast<float>(head_dim);
    xt::xarray<float> theta_pow = xt::pow(theta, expt_xt);

    auto freqs = xt::ones_like(theta_pow) / theta_pow;

    // Create sequence position tensor [0, 1, 2, ..., sequence_length-1] * head_dim
    xt::xarray<float> seq_pos = xt::arange<float>(sequence_length);
    xt::xarray<float> seq_pos_repeated_to_head = xt::repeat(seq_pos, head_dim, seq_pos.dimension() - 1);
    xt::xarray<float> scales = seq_pos_repeated_to_head.reshape({sequence_length, static_cast<uint32_t>(head_dim)});

    // scale the freqs by the sequence position
    xt::xarray<float> scaled_freqs = scales * freqs;

    // take the scaled freqs mod 2π to satisfy ttnn inputs constraints for sin/cos
    float pi = std::acos(-1.0F);
    scaled_freqs = xt::fmod(scaled_freqs, 2.0F * pi);
    scaled_freqs = scaled_freqs.reshape({1, 1, sequence_length, head_dim});

    xt::xarray<float> sin_freqs = xt::sin(scaled_freqs);
    xt::xarray<float> cos_freqs = xt::cos(scaled_freqs);

    auto* device = &autograd::ctx().get_device();
    return {core::from_xtensor(sin_freqs, device), core::from_xtensor(cos_freqs, device)};
}

ttnn::Tensor gen_trans_mat(int head_dim) {
    xt::xarray<float> trans_mat = xt::zeros<float>({1, 1, head_dim, head_dim});
    for (int i = 0; i < head_dim; i += 2) {
        trans_mat(0, 0, i, i + 1) = 1.0F;
    }
    for (int j = 1; j < head_dim; j += 2) {
        trans_mat(0, 0, j, j - 1) = -1.0F;
    }

    auto device = &autograd::ctx().get_device();
    return core::from_xtensor(trans_mat, device);
}

RotaryEmbeddingParams build_rope_params(uint32_t sequence_length, uint32_t head_dim, float theta) {
    if (head_dim % 32U != 0U) {
        throw std::invalid_argument("RoPE head_dim must be divisible by 32");
    }
    if (head_dim > 256U) {
        throw std::invalid_argument("RoPE head_dim must be less than or equal to 256");
    }
    if (head_dim <= 0U) {
        throw std::invalid_argument("RoPE head_dim must be greater than 0");
    }
    auto [sin_freqs, cos_freqs] = gen_freqs(head_dim, sequence_length, theta);
    auto trans_mat = gen_trans_mat(head_dim);

    return {
        .cos_cache = cos_freqs,
        .sin_cache = sin_freqs,
        .neg_cos_cache = cos_freqs,             // cos(θ) = cos(-θ): symmetry over x-axis
        .neg_sin_cache = ttnn::neg(sin_freqs),  // sin(-θ) = -sin(θ)
        .trans_mat = trans_mat,

        .sequence_length = sequence_length,
        .head_dim = head_dim,
    };
}

}  // namespace ttml::ops
