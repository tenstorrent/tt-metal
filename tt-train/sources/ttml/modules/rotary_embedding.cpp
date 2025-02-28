// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/rotary_embedding.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "core/xtensor_utils.hpp"
#include "ops/rope_op.hpp"

namespace ttml::modules {

RotaryEmbedding::RotaryEmbedding(RotaryEmbeddingParams& rope_params) : m_rope_params(rope_params) {
}

autograd::TensorPtr RotaryEmbedding::operator()(const autograd::TensorPtr& input) {
    return ttml::ops::rope(input, m_rope_params);
}

ttnn::Tensor composite_pow(ttnn::Tensor base, ttnn::Tensor expt) {
    // Compute base^expt using log and exp: b^k = e^(k * log(base))
    auto log_base = ttnn::log(base);
    auto expt_log_base = ttnn::experimental::mul(expt, log_base);  // handles broadcasting for us
    auto result = ttnn::exp(expt_log_base);                        // also handles broadcasting?
    return result;
}

ttnn::Tensor gen_freqs(uint32_t dim, uint32_t end, float theta = 10000.0F) {
    auto* device = &autograd::ctx().get_device();

    // Create a range tensor [0, 2, 4, ..., dim-2] and slice to dim/2
    std::vector<float> range_data;
    range_data.reserve(dim / 2);
    for (uint32_t i = 0; i < dim; i += 2) {
        if (i < dim / 2 * 2) {  // Ensure we only get dim/2 elements
            range_data.push_back(static_cast<float>(i));
        }
    }

    auto range_shape = core::create_shape({1, 1, 1, static_cast<uint32_t>(range_data.size())});
    auto range_tensor = core::from_vector(range_data, range_shape, device);

    // Compute freqs = 1.0 / (theta ** (range / dim))
    auto range_div_dim = ttnn::experimental::div(range_tensor, static_cast<float>(dim));
    ttnn::Tensor theta_pow =
        composite_pow(core::from_vector(std::vector{theta}, ttnn::Shape{1, 1, 1, 1}, device), range_div_dim);
    auto one_tensor = core::from_vector(std::vector{1.0F}, ttnn::Shape{1, 1, 1, 1}, device);
    auto freqs = ttnn::experimental::div(one_tensor, theta_pow);  // FIXME: we can probably collapse these

    // Create sequence position tensor [0, 1, 2, ..., end-1]
    std::vector<float> seq_pos_data;
    seq_pos_data.reserve(end);
    for (uint32_t i = 0; i < end; i++) {
        seq_pos_data.push_back(static_cast<float>(i));
    }

    auto seq_pos_shape = core::create_shape({1, 1, end, 1});
    auto seq_pos_tensor = core::from_vector(seq_pos_data, seq_pos_shape, device);

    auto scaled_freqs = ttnn::experimental::mul(seq_pos_tensor, freqs);

    return scaled_freqs;
}

ttnn::Tensor gen_trans_mat() {
    auto device = &autograd::ctx().get_device();
    xt::xarray<float> rot_emb_matrix = xt::zeros<float>({1, 1, 32, 32});
    // For even indices in the third dimension and odd indices in the fourth, set elements to 1.
    xt::view(rot_emb_matrix, xt::all(), xt::all(), xt::range(0, 32, 2), xt::range(1, 32, 2)) = 1.0f;
    // For odd indices in the third dimension and even indices in the fourth, set elements to -1.
    xt::view(rot_emb_matrix, xt::all(), xt::all(), xt::range(1, 32, 2), xt::range(0, 32, 2)) = -1.0f;
    auto rot_emb_matrix_tensor = core::from_xtensor(rot_emb_matrix, device);
    return rot_emb_matrix_tensor;
}

RotaryEmbeddingParams RotaryEmbedding::build_params(uint32_t sequence_length, uint32_t embedding_dim, float theta) {
    ttnn::Tensor freqs = gen_freqs(embedding_dim, sequence_length, theta);
    auto sin_freqs = ttnn::sin(freqs);
    auto cos_freqs = ttnn::cos(freqs);
    return {
        .cos_cache = cos_freqs,
        .sin_cache = sin_freqs,
        .neg_cos_cache = ttnn::neg(cos_freqs),
        .neg_sin_cache = ttnn::neg(sin_freqs),
        .trans_mat = gen_trans_mat(),
    };
}

}  // namespace ttml::modules
