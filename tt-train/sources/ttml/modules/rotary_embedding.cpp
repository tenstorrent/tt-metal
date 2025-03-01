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

ttnn::Tensor gen_freqs(uint32_t head_dim, uint32_t sequence_length, float theta = 10000.0F) {
    auto* device = &autograd::ctx().get_device();

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

    auto freqs = xt::ones_like(theta_pow) / theta_pow;  // FIXME: xt::reciprocal(theta_pow)?

    // Create sequence position tensor [0, 1, 2, ..., sequence_length-1] * head_dim
    xt::xarray<float> seq_pos = xt::arange<float>(sequence_length);
    xt::xarray<float> seq_pos_repeated_to_head = xt::repeat(seq_pos, head_dim, seq_pos.dimension() - 1);
    xt::xarray<float> scales = seq_pos_repeated_to_head.reshape({sequence_length, static_cast<uint32_t>(head_dim)});

    // scale the freqs by the sequence position
    xt::xarray<float> scaled_freqs = scales * freqs;

    return core::from_xtensor(scaled_freqs.reshape({1, 1, sequence_length, head_dim}), device);
}

ttnn::Tensor gen_trans_mat(int head_dim) {
    assert(head_dim % 32 == 0 && "head_dim must be divisible by 32");
    assert(head_dim <= 256 && "head_dim must be less than or equal to 256");
    assert(head_dim > 0 && "head_dim must be greater than 0");
    auto device = &autograd::ctx().get_device();
    xt::xarray<float> rot_emb_matrix = xt::zeros<float>({1, 1, head_dim, head_dim});
    // For even indices in the third dimension and odd indices in the fourth, set elements to 1.
    xt::view(rot_emb_matrix, xt::all(), xt::all(), xt::range(0, head_dim, 2), xt::range(1, head_dim, 2)) = 1.0f;
    // For odd indices in the third dimension and even indices in the fourth, set elements to -1.
    xt::view(rot_emb_matrix, xt::all(), xt::all(), xt::range(1, head_dim, 2), xt::range(0, head_dim, 2)) = -1.0f;
    auto rot_emb_matrix_tensor = core::from_xtensor(rot_emb_matrix, device);
    return rot_emb_matrix_tensor;
}

RotaryEmbeddingParams RotaryEmbedding::build_params(uint32_t sequence_length, uint32_t head_dim, float theta) {
    ttnn::Tensor freqs = gen_freqs(head_dim, sequence_length, theta);
    auto sin_freqs = ttnn::sin(freqs);
    auto cos_freqs = ttnn::cos(freqs);

    xt::xarray<float> expected_cos_freqs = {
        {1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F,
         1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F,
         1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F},
        {0.54030F, 0.54030F, 0.84601F, 0.84601F, 0.95042F, 0.95042F, 0.98423F, 0.98423F, 0.99500F, 0.99500F, 0.99842F,
         0.99842F, 0.99950F, 0.99950F, 0.99984F, 0.99984F, 0.99995F, 0.99995F, 0.99998F, 0.99998F, 0.99999F, 0.99999F,
         1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F},
        {-0.41615F, -0.41615F, 0.43146F, 0.43146F, 0.80658F, 0.80658F, 0.93742F, 0.93742F, 0.98007F, 0.98007F, 0.99368F,
         0.99368F,  0.99800F,  0.99800F, 0.99937F, 0.99937F, 0.99980F, 0.99980F, 0.99994F, 0.99994F, 0.99998F, 0.99998F,
         0.99999F,  0.99999F,  1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F, 1.00000F},
        {-0.98999F, -0.98999F, -0.11597F, -0.11597F, 0.58275F, 0.58275F, 0.86104F, 0.86104F,
         0.95534F,  0.95534F,  0.98580F,  0.98580F,  0.99550F, 0.99550F, 0.99858F, 0.99858F,
         0.99955F,  0.99955F,  0.99986F,  0.99986F,  0.99995F, 0.99995F, 0.99999F, 0.99999F,
         1.00000F,  1.00000F,  1.00000F,  1.00000F,  1.00000F, 1.00000F, 1.00000F, 1.00000F},
        {-0.65364F, -0.65364F, -0.62768F, -0.62768F, 0.30114F, 0.30114F, 0.75751F, 0.75751F,
         0.92106F,  0.92106F,  0.97481F,  0.97481F,  0.99201F, 0.99201F, 0.99747F, 0.99747F,
         0.99920F,  0.99920F,  0.99975F,  0.99975F,  0.99992F, 0.99992F, 0.99997F, 0.99997F,
         0.99999F,  0.99999F,  1.00000F,  1.00000F,  1.00000F, 1.00000F, 1.00000F, 1.00000F}};
    xt::xarray<float> expected_sin_freqs = {
        {0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F,
         0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F,
         0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F, 0.00000F},
        {0.84147F, 0.84147F, 0.53317F, 0.53317F, 0.31098F, 0.31098F, 0.17689F, 0.17689F, 0.09983F, 0.09983F, 0.05620F,
         0.05620F, 0.03162F, 0.03162F, 0.01778F, 0.01778F, 0.01000F, 0.01000F, 0.00562F, 0.00562F, 0.00316F, 0.00316F,
         0.00178F, 0.00178F, 0.00100F, 0.00100F, 0.00056F, 0.00056F, 0.00032F, 0.00032F, 0.00018F, 0.00018F},
        {0.90930F, 0.90930F, 0.90213F, 0.90213F, 0.59113F, 0.59113F, 0.34821F, 0.34821F, 0.19867F, 0.19867F, 0.11223F,
         0.11223F, 0.06320F, 0.06320F, 0.03556F, 0.03556F, 0.02000F, 0.02000F, 0.01125F, 0.01125F, 0.00632F, 0.00632F,
         0.00356F, 0.00356F, 0.00200F, 0.00200F, 0.00112F, 0.00112F, 0.00063F, 0.00063F, 0.00036F, 0.00036F},
        {0.14112F, 0.14112F, 0.99325F, 0.99325F, 0.81265F, 0.81265F, 0.50854F, 0.50854F, 0.29552F, 0.29552F, 0.16790F,
         0.16790F, 0.09473F, 0.09473F, 0.05332F, 0.05332F, 0.03000F, 0.03000F, 0.01687F, 0.01687F, 0.00949F, 0.00949F,
         0.00533F, 0.00533F, 0.00300F, 0.00300F, 0.00169F, 0.00169F, 0.00095F, 0.00095F, 0.00053F, 0.00053F},
        {-0.75680F, -0.75680F, 0.77847F, 0.77847F, 0.95358F, 0.95358F, 0.65283F, 0.65283F, 0.38942F, 0.38942F, 0.22304F,
         0.22304F,  0.12615F,  0.12615F, 0.07107F, 0.07107F, 0.03999F, 0.03999F, 0.02249F, 0.02249F, 0.01265F, 0.01265F,
         0.00711F,  0.00711F,  0.00400F, 0.00400F, 0.00225F, 0.00225F, 0.00126F, 0.00126F, 0.00071F, 0.00071F}};

    // they look approx ok ✓
    /*     if (!xt::allclose(core::to_xtensor(cos_freqs), expected_cos_freqs)) {
            std::cout << "cos_freqs: " << core::to_xtensor(cos_freqs) << std::endl;
            std::cout << "expected_cos_freqs: " << expected_cos_freqs << std::endl;
        }
        if (!xt::allclose(core::to_xtensor(sin_freqs), expected_sin_freqs)) {
            std::cout << "sin_freqs: " << core::to_xtensor(sin_freqs) << std::endl;
            std::cout << "expected_sin_freqs: " << expected_sin_freqs << std::endl;
        } */

    return {
        .cos_cache = cos_freqs,
        .sin_cache = sin_freqs,
        .neg_cos_cache = cos_freqs,             // cos(θ) = cos(-θ): symmetry over x-axis
        .neg_sin_cache = ttnn::neg(sin_freqs),  // sin(-θ) = -sin(θ)
        .trans_mat = gen_trans_mat(head_dim),
    };
}

}  // namespace ttml::modules
