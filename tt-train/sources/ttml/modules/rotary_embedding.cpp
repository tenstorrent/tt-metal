// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/rotary_embedding.hpp"

#include <xtensor/xmanipulation.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "core/xtensor_utils.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

RotaryEmbedding::RotaryEmbedding(const ops::RotaryEmbeddingParams& rope_params) : m_rope_params(rope_params) {
}

autograd::TensorPtr RotaryEmbedding::operator()(const autograd::TensorPtr& input) {
    return ttml::ops::rope(input, m_rope_params);
}

ttnn::Tensor from_xtensor_to_l1(const xt::xarray<float>& x) {
    auto device = &autograd::ctx().get_device();
    if (x.dimension() != 4) {
        throw std::invalid_argument("x must have 4 dimensions");
    }
    auto x_tt_tensor = core::from_xtensor(x, device);
    return ttnn::to_memory_config(x_tt_tensor, ttnn::L1_MEMORY_CONFIG);
}

ttnn::Tensor gen_freqs(uint32_t head_dim, uint32_t sequence_length, float theta = 10000.0F) {
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
    scaled_freqs = xt::fmod(scaled_freqs, 2.0F * 3.14159265358979323846F);

    return from_xtensor_to_l1(scaled_freqs.reshape({1, 1, sequence_length, head_dim}));
}

ttnn::Tensor gen_trans_mat(int head_dim) {
    xt::xarray<float> trans_mat = xt::zeros<float>({1, 1, head_dim, head_dim});
    for (int i = 0; i < head_dim; i += 2) {
        trans_mat(0, 0, i, i + 1) = 1.0F;
    }
    for (int j = 1; j < head_dim; j += 2) {
        trans_mat(0, 0, j, j - 1) = -1.0F;
    }
    return from_xtensor_to_l1(trans_mat);
}

ops::RotaryEmbeddingParams RotaryEmbedding::build_params(uint32_t sequence_length, uint32_t head_dim, float theta) {
    if (head_dim % 32 != 0) {
        throw std::invalid_argument("RoPE head_dim must be divisible by 32");
    }
    if (head_dim > 256) {
        throw std::invalid_argument("RoPE head_dim must be less than or equal to 256");
    }
    if (head_dim <= 0) {
        throw std::invalid_argument("RoPE head_dim must be greater than 0");
    }
    ttnn::Tensor freqs = gen_freqs(head_dim, sequence_length, theta);
    auto sin_freqs = ttnn::sin(freqs);
    auto cos_freqs = ttnn::cos(freqs);
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

}  // namespace ttml::modules
