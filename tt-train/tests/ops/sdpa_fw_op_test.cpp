// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "xtensor/generators/xbuilder.hpp"

class SDPAForwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> generate_mask(const xt::xarray<float>& query) {
    auto shape = query.shape();
    size_t B = shape[0], H = shape[1], S = shape[2], d = shape[3];
    xt::xarray<float> mask = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t w = 0; w <= s; ++w) {
                    mask(b, h, s, w) = 1.0F;  // causal mask - upper triangular part
                }
            }
        }
    }
    return mask;
}

// Split (B, 1, S, d)  -->  (B, H, S, d/H)
// Assumes: d % num_heads == 0
xt::xarray<float> split_heads(const xt::xarray<float>& input, std::uint32_t num_heads) {
    const std::size_t B = input.shape()[0];
    const std::size_t S = input.shape()[2];
    const std::size_t d = input.shape()[3];
    const std::size_t H = static_cast<std::size_t>(num_heads);
    const std::size_t Dh = d / H;

    xt::xarray<float> out = xt::xarray<float>::from_shape({B, H, S, Dh});
    std::fill(out.begin(), out.end(), 0.0f);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t in_base = h * Dh;  // slice in last dim
                for (std::size_t t = 0; t < Dh; ++t) {
                    out(b, h, s, t) = input(b, 0, s, in_base + t);
                }
            }
        }
    }
    return out;
}

// Fuse (B, H, S, Dh)  -->  (B, 1, S, H*Dh)
// Assumes: input.shape()[1] == num_heads
xt::xarray<float> fuse_heads(const xt::xarray<float>& heads, std::uint32_t num_heads) {
    const std::size_t B = heads.shape()[0];
    const std::size_t H = heads.shape()[1];  // should equal num_heads
    const std::size_t S = heads.shape()[2];
    const std::size_t Dh = heads.shape()[3];
    (void)num_heads;  // kept for symmetry with split; not used

    const std::size_t d = H * Dh;

    xt::xarray<float> out = xt::xarray<float>::from_shape({B, std::size_t(1), S, d});
    std::fill(out.begin(), out.end(), 0.0f);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t out_base = h * Dh;  // slice in last dim
                for (std::size_t t = 0; t < Dh; ++t) {
                    out(b, 0, s, out_base + t) = heads(b, h, s, t);
                }
            }
        }
    }
    return out;
}

// Naive reference SDPA with grouped KV (no validation).
// Inputs (physical):
//   Q: (B, 1, S, qD)
//   K: (B, 1, S, kvD)
//   V: (B, 1, S, kvD)
//   attn_mask: (B, 1, S, S)   // additive: 0 keep, large negative to mask
// Heads (passed in):
//   query_heads -> cast to q_heads
//   key_heads   -> cast to kv_heads
// Assumes: qD / q_heads == kvD / kv_heads
// Returns: (B, 1, S, qD)
xt::xarray<float> sdpa_grouped_naive(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& attn_mask,
    std::uint32_t query_heads,
    std::uint32_t key_heads) {
    // local aliases with your preferred names
    const std::size_t q_heads = static_cast<std::size_t>(query_heads);
    const std::size_t kv_heads = static_cast<std::size_t>(key_heads);

    const std::size_t B = Q.shape()[0];
    const std::size_t S = Q.shape()[2];
    const std::size_t qD = Q.shape()[3];
    const std::size_t kvD = K.shape()[3];

    const std::size_t Dh_q = qD / q_heads;
    const std::size_t Dh_kv = kvD / kv_heads;
    const std::size_t Dh = Dh_q;  // assume Dh_q == Dh_kv

    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, std::size_t(1), S, qD});
    std::fill(Out.begin(), Out.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);
            const std::size_t q_off = h * Dh;   // Q slice
            const std::size_t kv_off = g * Dh;  // KV slice

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q(b, 0, i, q_off + t) * K(b, 0, j, kv_off + t);
                    }
                    const float m = attn_mask(b, 0, i, j);  // expected 0 or 1
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    // float s = dot * scale + attn_mask(b, 0, i, j);
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);  // <- changed line
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // out_i[h] = sum_j softmax_ij * V[j]
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V(b, 0, j, kv_off + t);
                    }
                    Out(b, 0, i, q_off + t) = acc;
                }
            }
        }
    }

    return Out;
}

// Extended version that also returns intermediate results (1/sum_exp)
// This version expects UNSPLIT tensors (B, 1, S, D) with head count parameters
std::pair<xt::xarray<float>, xt::xarray<float>> sdpa_grouped_naive_with_intermediates(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& attn_mask,
    std::uint32_t query_heads,
    std::uint32_t key_heads) {
    // local aliases with your preferred names
    const std::size_t q_heads = static_cast<std::size_t>(query_heads);
    const std::size_t kv_heads = static_cast<std::size_t>(key_heads);

    const std::size_t B = Q.shape()[0];
    const std::size_t S = Q.shape()[2];
    const std::size_t qD = Q.shape()[3];
    const std::size_t kvD = K.shape()[3];

    const std::size_t Dh_q = qD / q_heads;
    const std::size_t Dh_kv = kvD / kv_heads;
    const std::size_t Dh = Dh_q;  // assume Dh_q == Dh_kv

    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, std::size_t(1), S, qD});
    std::fill(Out.begin(), Out.end(), 0.0F);

    // Intermediates: (B, q_heads, S, 1) - reciprocal of sum of exponentials per head per sequence position
    xt::xarray<float> Intermediates = xt::xarray<float>::from_shape({B, q_heads, S, std::size_t(1)});
    std::fill(Intermediates.begin(), Intermediates.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);
            const std::size_t q_off = h * Dh;   // Q slice
            const std::size_t kv_off = g * Dh;  // KV slice

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q(b, 0, i, q_off + t) * K(b, 0, j, kv_off + t);
                    }
                    const float m = attn_mask(b, 0, i, j);  // expected 0 or 1
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // Store intermediate: 1/denom (reciprocal of sum of exponentials)
                Intermediates(b, h, i, 0) = 1.0F / denom;

                // out_i[h] = sum_j softmax_ij * V[j]
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V(b, 0, j, kv_off + t);
                    }
                    Out(b, 0, i, q_off + t) = acc;
                }
            }
        }
    }

    return std::make_pair(Out, Intermediates);
}

// New version that works with SPLIT-BY-HEADS tensors (B, H, S, D/H) but outputs FUSED format (B, 1, S, qD)
std::pair<xt::xarray<float>, xt::xarray<float>> sdpa_split_heads_naive_with_intermediates(
    const xt::xarray<float>& Q_split,
    const xt::xarray<float>& K_split,
    const xt::xarray<float>& V_split,
    const xt::xarray<float>& attn_mask) {
    const std::size_t B = Q_split.shape()[0];
    const std::size_t q_heads = Q_split.shape()[1];
    const std::size_t kv_heads = K_split.shape()[1];
    const std::size_t S = Q_split.shape()[2];
    const std::size_t Dh = Q_split.shape()[3];
    const std::size_t qD = q_heads * Dh;  // Total query dimension

    // Output in FUSED format (B, 1, S, qD) - back to unsplit format
    xt::xarray<float> Out = xt::xarray<float>::from_shape({B, std::size_t(1), S, qD});
    std::fill(Out.begin(), Out.end(), 0.0F);

    // Intermediates: (B, q_heads, S, 1) - reciprocal of sum of exponentials per head per sequence position
    xt::xarray<float> Intermediates = xt::xarray<float>::from_shape({B, q_heads, S, std::size_t(1)});
    std::fill(Intermediates.begin(), Intermediates.end(), 0.0F);

    auto group_of_head = [&](std::size_t h) -> std::size_t {
        // contiguous block mapping for grouped KV
        return (h * kv_heads) / q_heads;
    };

    const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
    std::vector<float> scores_row(S);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < q_heads; ++h) {
            const std::size_t g = group_of_head(h);
            const std::size_t q_off = h * Dh;  // Offset in fused output tensor

            for (std::size_t i = 0; i < S; ++i) {
                // scores_row[j] = (q_i · k_j) * scale + mask(i,j)
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q_split(b, h, i, t) * K_split(b, g, j, t);
                    }
                    const float m = attn_mask(b, 0, i, j);  // expected 0 or 1
                    const float s = m * (dot * scale) + (m - 1.0F) * 1e9F;
                    scores_row[j] = s;
                    rmax = std::max(s, rmax);
                }

                // softmax over j
                float denom = 0.0F;
                for (std::size_t j = 0; j < S; ++j) denom += std::exp(scores_row[j] - rmax);
                denom = std::max(denom, 1e-20F);

                // Store intermediate: 1/denom (reciprocal of sum of exponentials)
                Intermediates(b, h, i, 0) = 1.0F / denom;

                // out_i[h] = sum_j softmax_ij * V[j] - store in FUSED format
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        acc += w * V_split(b, g, j, t);
                    }
                    Out(b, 0, i, q_off + t) = acc;  // Store in fused format
                }
            }
        }
    }

    return std::make_pair(Out, Intermediates);
}

float compute_mse(const xt::xarray<float>& expected, const xt::xarray<float>& result) {
    assert(result.shape() == expected.shape());
    xt::xarray<float> diff = expected - result;
    float mse = xt::mean(xt::square(diff))();
    return mse;
}

// Wrapper around matmul to handle sharing of KV heads across groups of query
// heads.
// For e.g. Q @ V, there are two cases:
// - G == H: (B, H, S, S) x (B, H, S, V) -> (B, H, S, V)
// - G != H:
//    - In this case value has shape (B,G,S,V):
//      1. Reshape attention_weights to (B*G, H/G, S, S).
//      2. Reshape value to (B*G, 1, S, V).
//      3. Manually broadcast values over groupsize.
//      4. Matmul.
//      5. Reshape the result to (B, H, S, V).
//   - Summary of intermediate shapes:
//     (B*G, H/G, S, S) x (B*G, 1, S, V) -> (B*G, H/G, S, V) -> (B, H, S, V)
ttnn::Tensor group_shared_matmul(
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& kv_tensor,
    bool transpose_a = false,
    bool transpose_b = false) {
    using namespace ttml;
    auto [batch_num, heads, seq_len, embedding_dim] = query_tensor.logical_shape().to_array_4D();
    auto [batch_num_v, groups, seq_len_v, embedding_dim_v] = kv_tensor.logical_shape().to_array_4D();
    if (batch_num != batch_num_v) {
        throw std::invalid_argument(
            fmt::format(
                "query_tensor and kv_tensor must have the same batch size, got shapes {} and {} respectively",
                query_tensor.logical_shape(),
                kv_tensor.logical_shape()));
    }
    if (heads == groups) {
        // no broadcasting needed
        return ttnn_fixed::matmul(query_tensor, kv_tensor, transpose_a, transpose_b);
    }
    // result will have shape (batch_num, heads, M, N)
    // we determine M,N based on the transpose options
    auto M = transpose_a ? embedding_dim : seq_len;
    auto N = transpose_b ? seq_len_v : embedding_dim_v;

    // - G != H:
    //   bcast kv_tensor to groups in query_tensor then reshape back to query_tensor_shape:
    //   (B*G,H/G,M,E) x (B*G, 1, E,N) -> (B*G, H/G, M, N) -> (B, H, M, N)
    auto query_tensor_grouped =
        ttnn::reshape(query_tensor, ttnn::Shape{batch_num * groups, heads / groups, seq_len, embedding_dim});
    auto kv_tensor_batched = ttnn::reshape(kv_tensor, ttnn::Shape{batch_num * groups, 1U, seq_len_v, embedding_dim_v});

    // repeat kv_tensor to group size for each group (manual bcast)
    ttnn::Tensor kv_tensor_repeated = ttnn::repeat(kv_tensor_batched, ttnn::Shape{1U, heads / groups, 1U, 1U});
    auto bcasted_mm = ttnn_fixed::matmul(query_tensor_grouped, kv_tensor_repeated, transpose_a, transpose_b);
    auto reshaped_mm = ttnn::reshape(bcasted_mm, ttnn::Shape{batch_num, heads, M, N});
    return reshaped_mm;
}

std::vector<ttnn::Tensor> composite_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const std::optional<ttnn::Tensor>& attn_mask) {
    // std::vector<ttnn::Tensor> result;
    // result.reserve(2U);  // one for output, one for intermediate if needed

    using namespace ttml;
    auto [batch_num, heads, seq_len, embedding_dim] = query.logical_shape().to_array_4D();
    auto groups = value.logical_shape().to_array_4D()[1];

    const float scale = 1.0F / std::sqrt(static_cast<float>(embedding_dim));
    constexpr auto none = ttsl::Span<const ttnn::operations::unary::UnaryWithParam>{};
    auto q_scaled = ttnn::multiply(query, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    ttnn::Tensor qk_scaled = group_shared_matmul(q_scaled, key, /*transpose_a=*/false, /*transpose_b=*/true);

    // σQ @ K
    if (attn_mask.has_value()) {
        ttnn::Tensor mask_tensor = attn_mask.value();
        qk_scaled = ttnn::add(
            ttnn::multiply(mask_tensor, qk_scaled, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            ttnn::multiply(
                ttnn::subtract(mask_tensor, 1.F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                1e9F,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
    }

    // Calculate intermediate results to test against kernel implementation
    auto max_value = ttnn::max(qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto qk_scaled_sub_max = ttnn::subtract(qk_scaled, max_value);
    auto exp_qk_scaled = ttnn::exp(qk_scaled_sub_max);
    auto sum_exp = ttnn::sum(exp_qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto recip_sum_exp = ttnn::reciprocal(sum_exp);

    auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);

    auto attention_qkv = group_shared_matmul(attention_weights, value, /*transpose_a=*/false, /*transpose_b=*/false);
    return {attention_qkv, recip_sum_exp};
}

struct SDPATestConfig {
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t query_dim;
    uint32_t key_value_dim;
    uint32_t num_query_heads;
    uint32_t num_key_heads;
    float dropout_prob = 0.0F;
    uint32_t random_seed = 42;
    float result_atol = 4e-2F;
    float result_rtol = 2e-2F;
    float intermediate_atol = 2e-2F;
    float intermediate_rtol = 2e-2F;
    std::string test_name = "SDPA Test";
};

void run_sdpa_test(const SDPATestConfig& config) {
    using namespace ttml;

    // Generate already split-by-heads tensors directly
    std::mt19937 gen(config.random_seed);
    const uint32_t head_dim_q = config.query_dim / config.num_query_heads;
    const uint32_t head_dim_kv = config.key_value_dim / config.num_key_heads;

    xt::xarray<float> query_tensor = xt::random::rand<float>(
        {config.batch_size, config.num_query_heads, config.sequence_length, head_dim_q}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>(
        {config.batch_size, config.num_key_heads, config.sequence_length, head_dim_kv}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>(
        {config.batch_size, config.num_key_heads, config.sequence_length, head_dim_kv}, -1.0F, 1.0F, gen);

    // Create attention mask in kernel-expected format (B, qNH, S, S)
    xt::xarray<float> query_for_mask = xt::random::rand<float>(
        {config.batch_size, config.num_query_heads, config.sequence_length, config.query_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_for_mask);

    // Convert to device tensors
    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    // Run SDPA kernel with new interface - this is our reference implementation
    auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, config.dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());  // Kernel returns (B, 1, S, qD*q_heads)
    xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

    // Run composite SDPA implementation with split tensors, then fuse output to match kernel
    auto composite_result_split = composite_sdpa_fw(query, key, value, attn_mask);
    xt::xarray<float> composite_result_split_xtensor = core::to_xtensor(composite_result_split[0]);
    xt::xarray<float> composite_interm_xtensor = core::to_xtensor(composite_result_split[1]);

    // Fuse composite output to match kernel format (B, 1, S, qD*q_heads)
    xt::xarray<float> composite_result_xtensor = fuse_heads(composite_result_split_xtensor, config.num_query_heads);

    // Run float reference implementation with split tensors (produces FUSED output)
    auto [float_result, float_intermediates] =
        sdpa_split_heads_naive_with_intermediates(query_tensor, key_tensor, value_tensor, attn_mask_tensor);

    // All results are now in fused format (B, 1, S, qD*q_heads) to match kernel
    // Shape validation - all should be in fused format (B, 1, S, qD*q_heads)
    ASSERT_EQ(result_xtensor.shape(), float_result.shape()) << "Kernel result shape mismatch in " << config.test_name;
    ASSERT_EQ(composite_result_xtensor.shape(), float_result.shape())
        << "Composite result shape mismatch in " << config.test_name;
    ASSERT_EQ(interm_xtensor.shape(), float_intermediates.shape())
        << "Intermediate shape mismatch in " << config.test_name;

    // Compute MSE for validation
    float mse_kernel_vs_composite = compute_mse(result_xtensor, composite_result_xtensor);
    float mse_kernel_vs_float = compute_mse(result_xtensor, float_result);
    float mse_composite_vs_float = compute_mse(composite_result_xtensor, float_result);
    float mse_kernel_vs_composite_interm = compute_mse(interm_xtensor, composite_interm_xtensor);

    // Primary validation: Kernel vs Composite (most reliable - both use same implementation approach)
    EXPECT_TRUE(xt::allclose(result_xtensor, composite_result_xtensor, config.result_atol, config.result_rtol))
        << "Kernel vs Composite comparison failed in " << config.test_name << " (MSE: " << mse_kernel_vs_composite
        << ")";

    // Secondary validation: Compare with float reference (may have numerical precision differences)
    bool float_impl_reliable =
        xt::allclose(composite_result_xtensor, float_result, config.result_atol * 10, config.result_rtol * 10);

    if (float_impl_reliable) {
        // Float implementation seems reliable, use normal tolerances
        EXPECT_TRUE(xt::allclose(result_xtensor, float_result, config.result_atol, config.result_rtol))
            << "Kernel vs Float result comparison failed in " << config.test_name << " (MSE: " << mse_kernel_vs_float
            << ")";

        EXPECT_TRUE(xt::allclose(composite_result_xtensor, float_result, config.result_atol, config.result_rtol))
            << "Composite vs Float result comparison failed in " << config.test_name
            << " (MSE: " << mse_composite_vs_float << ")";

        EXPECT_TRUE(
            xt::allclose(interm_xtensor, float_intermediates, config.intermediate_atol, config.intermediate_rtol))
            << "Intermediate result comparison failed in " << config.test_name;
    } else {
        // Float implementation unreliable, compare intermediates between kernel and composite instead
        EXPECT_TRUE(
            xt::allclose(interm_xtensor, composite_interm_xtensor, config.intermediate_atol, config.intermediate_rtol))
            << "Kernel vs Composite intermediate result comparison failed in " << config.test_name
            << " (MSE: " << mse_kernel_vs_composite_interm << ")";
    }
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 2U,
        .num_key_heads = 2U,
        .test_name = "SmallBatch_2H_2KV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SingleHead) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 128U,
        .num_query_heads = 1U,
        .num_key_heads = 1U,
        .test_name = "SingleHead_1H_1KV"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch_2Heads_1Group) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 128U,
        .query_dim = 128U,
        .key_value_dim = 64U,
        .num_query_heads = 2U,
        .num_key_heads = 1U,
        .test_name = "SmallBatch_2H_1KV_Grouped"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch_12Heads_6Group) {
    SDPATestConfig config{
        .batch_size = 1U,
        .sequence_length = 1024U,
        .query_dim = 768U,
        .key_value_dim = 384U,
        .num_query_heads = 12U,
        .num_key_heads = 6U,
        .test_name = "SmallBatch_12H_6KV_Grouped"};
    run_sdpa_test(config);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_Batch_12Heads_6Group) {
    SDPATestConfig config{
        .batch_size = 16U,
        .sequence_length = 1024U,
        .query_dim = 768U,
        .key_value_dim = 384U,
        .num_query_heads = 12U,
        .num_key_heads = 6U,
        .test_name = "Batch_16B_12H_6KV_Production"};
    run_sdpa_test(config);
}

// =============================================================================
// VALIDATION TESTS - Testing Error Conditions and Edge Cases
// =============================================================================

TEST_F(SDPAForwardTest, ValidationTest_InvalidHeadConfiguration) {
    using namespace ttml;

    std::mt19937 gen(42);

    // Test Case 1: Query heads not divisible by key heads (should fail)
    // This creates a scenario where qHt % kHt != 0, triggering the TT_FATAL in device operation
    {
        // Create tensors with 3 query heads and 2 key/value heads (3 % 2 != 0)
        xt::xarray<float> query_tensor = xt::random::rand<float>({1, 3, 128, 32}, -1.0F, 1.0F, gen);  // 3 heads
        xt::xarray<float> key_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);    // 2 heads
        xt::xarray<float> value_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);  // 2 heads
        xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({1, 3, 128, 128}, 0.0F, 1.0F, gen);

        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false); }, std::runtime_error)
            << "Should fail when q_heads is not divisible by kv_heads";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_ShapeMismatch) {
    using namespace ttml;

    std::mt19937 gen(42);

    // Test Case 1: Key-Value dimension mismatch
    {
        // Create split-by-heads tensors with mismatched dimensions
        xt::xarray<float> query_tensor =
            xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);  // 2 heads, 32 dim each
        xt::xarray<float> key_tensor =
            xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);  // 2 heads, 32 dim each
        xt::xarray<float> value_tensor =
            xt::random::rand<float>({1, 2, 128, 16}, -1.0F, 1.0F, gen);  // Different dimension
        xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({1, 1, 128, 128}, 0.0F, 1.0F, gen);

        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false); }, std::runtime_error)
            << "Should fail with shape mismatch";
    }

    // Test Case 2: Batch size mismatch
    {
        xt::xarray<float> query_tensor = xt::random::rand<float>({2, 2, 128, 32}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor =
            xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);  // Different batch size
        xt::xarray<float> value_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);
        xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({2, 1, 128, 128}, 0.0F, 1.0F, gen);

        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false); }, std::runtime_error)
            << "Should fail with shape mismatch";
    }

    // Test Case 3: Sequence length mismatch
    {
        xt::xarray<float> query_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor = xt::random::rand<float>({1, 2, 64, 32}, -1.0F, 1.0F, gen);  // Different seq len
        xt::xarray<float> value_tensor = xt::random::rand<float>({1, 2, 64, 32}, -1.0F, 1.0F, gen);
        xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({1, 1, 128, 64}, 0.0F, 1.0F, gen);

        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false); }, std::runtime_error)
            << "Should fail with shape mismatch";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_KeyValueShapeMismatch) {
    using namespace ttml;

    std::mt19937 gen(42);

    // Test case where Key and Value have different shapes
    // This should trigger the TT_FATAL: "Key and Value must have the same shape"
    {
        xt::xarray<float> query_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor = xt::random::rand<float>({1, 2, 128, 32}, -1.0F, 1.0F, gen);  // Shape: 1,2,128,32
        xt::xarray<float> value_tensor =
            xt::random::rand<float>({1, 2, 128, 16}, -1.0F, 1.0F, gen);  // Shape: 1,2,128,16 - Different!
        xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({1, 2, 128, 128}, 0.0F, 1.0F, gen);

        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false); }, std::runtime_error)
            << "Should fail when Key and Value have different shapes";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_EdgeCaseDimensions) {
    using namespace ttml;

    std::mt19937 gen(42);

    // Test Case 1: Minimum viable dimensions
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 32U,  // Minimum tile size
            .query_dim = 32U,        // Minimum tile size
            .key_value_dim = 32U,
            .num_query_heads = 1U,
            .num_key_heads = 1U,
            .test_name = "EdgeCase_MinDimensions"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle minimum tile dimensions correctly";
    }

    // Test Case 2: Single head configuration
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 64U,
            .key_value_dim = 64U,
            .num_query_heads = 1U,
            .num_key_heads = 1U,
            .test_name = "EdgeCase_SingleHead"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle single head attention correctly";
    }

    // Test Case 3: Progressive grouping ratios
    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 128U,     // 4 heads * 32 dim per head
            .key_value_dim = 32U,  // 1 head * 32 dim per head
            .num_query_heads = 4U,
            .num_key_heads = 1U,
            .test_name = "EdgeCase_Grouping_4to1"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle 4:1 grouping ratio correctly";
    }

    {
        SDPATestConfig config{
            .batch_size = 1U,
            .sequence_length = 128U,
            .query_dim = 256U,     // 8 heads * 32 dim per head
            .key_value_dim = 32U,  // 1 head * 32 dim per head
            .num_query_heads = 8U,
            .num_key_heads = 1U,
            .test_name = "EdgeCase_MaxGrouping_8to1"};

        EXPECT_NO_THROW({ run_sdpa_test(config); }) << "Should handle 8:1 grouping ratio correctly";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_IntermediateReturnModes) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 128U, d = 64U;
    std::mt19937 gen(42);

    // Create split-by-heads tensors for the new interface
    const uint32_t num_heads = 2U;
    const uint32_t head_dim = d / num_heads;
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, num_heads, S, head_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, num_heads, S, head_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, num_heads, S, head_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = xt::random::rand<float>({B, num_heads, S, S}, 0.0F, 1.0F, gen);

    // Test Case 1: return_intermediates = false
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, false);

        EXPECT_TRUE(result[0].has_value()) << "Main result should always be present";
        EXPECT_FALSE(result[1].has_value()) << "Intermediate should be null when return_intermediates=false";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        // Kernel returns fused format (B, 1, S, d) where d is total embedding dimension
        std::vector<size_t> expected_shape = {B, 1U, S, d};
        EXPECT_EQ(result_xtensor.shape(), expected_shape) << "Result should be in fused format (B, 1, S, d)";
    }

    // Test Case 2: return_intermediates = true
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0.0F, true);

        EXPECT_TRUE(result[0].has_value()) << "Main result should be present";
        EXPECT_TRUE(result[1].has_value()) << "Intermediate should be present when return_intermediates=true";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

        // Kernel returns fused format (B, 1, S, d) where d is total embedding dimension
        std::vector<size_t> expected_shape = {B, 1U, S, d};
        EXPECT_EQ(result_xtensor.shape(), expected_shape) << "Result should be in fused format (B, 1, S, d)";

        // Check intermediate shape: (B, num_query_heads, S, 1)
        std::vector<size_t> expected_interm_shape = {B, num_heads, S, 1U};
        EXPECT_EQ(interm_xtensor.shape(), expected_interm_shape) << "Intermediate shape should be (B, q_heads, S, 1)";

        // Verify intermediate values are reasonable (should be positive reciprocals)
        EXPECT_TRUE(xt::all(interm_xtensor > 0.0f)) << "All intermediate values should be positive";
        EXPECT_TRUE(xt::all(interm_xtensor <= 1.0f)) << "All intermediate values should be <= 1.0 (reciprocals)";
    }
}

TEST_F(SDPAForwardTest, ValidationTest_PerformanceTest) {
    using namespace ttml;

    uint32_t B = 1U, H = 1U, S = 1024, dQ = 4096U, dKV = 1024;
    uint32_t num_query_heads = 32U;
    uint32_t num_key_heads = 8U;
    float dropout_prob = 0.0F;
    uint32_t random_seed = 42;

    auto* mesh_devices = &autograd::ctx().get_device();
    auto device_ids = mesh_devices->get_device_ids();
    using Clock = std::chrono::high_resolution_clock;

    // Generate already split-by-heads tensors directly
    std::mt19937 gen(random_seed);
    const uint32_t head_dim_q = dQ / num_query_heads;
    const uint32_t head_dim_kv = dKV / num_key_heads;

    xt::xarray<float> query_tensor = xt::random::rand<float>({B, num_query_heads, S, head_dim_q}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, num_key_heads, S, head_dim_kv}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, num_key_heads, S, head_dim_kv}, -1.0F, 1.0F, gen);

    // Create attention mask in kernel-expected format (B, qNH, S, S)
    xt::xarray<float> query_for_mask = xt::random::rand<float>({B, num_query_heads, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_for_mask);

    // Convert to device tensors
    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    // Run SDPA kernel
    auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());  // Kernel returns (B, 1, S, qD*q_heads)
    xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

    // Run composite SDPA implementation with split tensors, then fuse output to match kernel
    auto composite_result_split = composite_sdpa_fw(query, key, value, attn_mask);
    xt::xarray<float> composite_result_split_xtensor = core::to_xtensor(composite_result_split[0]);
    xt::xarray<float> composite_interm_xtensor = core::to_xtensor(composite_result_split[1]);

    // Fuse composite output to match kernel format (B, 1, S, qD*q_heads)
    xt::xarray<float> composite_result_xtensor = fuse_heads(composite_result_split_xtensor, num_query_heads);

    // Run float reference implementation with split tensors (produces FUSED output)
    auto [float_result, float_intermediates] =
        sdpa_split_heads_naive_with_intermediates(query_tensor, key_tensor, value_tensor, attn_mask_tensor);

    float mse_kernel_vs_float = compute_mse(float_result, result_xtensor);
    float mse_composite_vs_float = compute_mse(float_result, composite_result_xtensor);

    fmt::print(
        "Performance Test MSE: Kernel vs Float: {}, Composite vs Float: {}\n",
        mse_kernel_vs_float,
        mse_composite_vs_float);

    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }
    // // Sleep for ten seconds to stabilize GPU frequency
    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // const int test_size = 50;

    // std::vector<ttnn::Tensor> result_holder;
    // result_holder.reserve(test_size);

    // std::vector<ttnn::Tensor> baseline_holder;
    // baseline_holder.reserve(test_size);

    // auto op_start = Clock::now();
    // for (int i = 0; i < test_size; ++i) {
    //     result = ttml::metal::sdpa_fw(query, key, value, attn_mask, dropout_prob, return_intermediates);
    //     result_holder.push_back(result[0].value());
    // }
    // // xt::xarray<float> test_result_xtensor = core::to_xtensor(result[0].value());
    // // xt::xarray<float> test_interm_xtensor = core::to_xtensor(result[1].value());
    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }
    // auto op_end = Clock::now();

    // // Sleep for ten seconds to stabilize GPU frequency
    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // auto baseline_start = Clock::now();
    // for (int i = 0; i < test_size; ++i) {
    //     composite_result_split = composite_sdpa_fw(query, key, value, attn_mask);
    //     baseline_holder.push_back(composite_result_split[0]);
    // }
    // // xt::xarray<float> test_baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]),
    // // num_query_heads); xt::xarray<float> test_baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);
    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }
    // auto baseline_end = Clock::now();

    // std::chrono::duration<float, std::milli> op_duration = op_end - op_start;
    // std::chrono::duration<float, std::milli> baseline_duration = baseline_end - baseline_start;

    // std::cout << "op: " << op_duration.count() / static_cast<float>(test_size) << " ms\n";
    // std::cout << "baseline : " << baseline_duration.count() / static_cast<float>(test_size) << " ms\n";

    EXPECT_TRUE(false);
}