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

xt::xarray<float> dot(const xt::xarray<float>& A, const xt::xarray<float>& B) {
    assert(A.dimension() == 2 && B.dimension() == 2);
    assert(A.shape()[1] == B.shape()[0]);  // A: (M, K), B: (K, N)

    const size_t M = A.shape()[0];
    const size_t K = A.shape()[1];
    const size_t N = B.shape()[1];

    xt::xarray<float> result = xt::zeros<float>({M, N});

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

xt::xarray<float> generate_mask(const xt::xarray<float>& query) {
    auto shape = query.shape();
    size_t B = shape[0], H = shape[1], S = shape[2], d = shape[3];
    xt::xarray<float> mask = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < S; ++s) {
                for (size_t w = 0; w <= s; ++w) {
                    mask(b, h, s, w) = 1.0F;  // upper triangular part
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
    // auto intm_result = ttnn_fixed::matmul(exp_qk_scaled, value, /*transpose_a=*/false, /*transpose_b=*/false);
    auto sum_exp = ttnn::sum(exp_qk_scaled, /* dim */ 3, /* keepdim */ true);
    auto recip_sum_exp = ttnn::reciprocal(sum_exp);
    // auto interm_mm_result = ttnn_fixed::matmul(exp_qk_scaled, value, /*transpose_a=*/false, /*transpose_b=*/false);

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

    const uint32_t H = 1U;  // Always 1 in current tests
    fmt::print(
        "=== Running {} (B={}, S={}, qH={}, kvH={}) ===\n",
        config.test_name,
        config.batch_size,
        config.sequence_length,
        config.num_query_heads,
        config.num_key_heads);

    // Generate test data
    std::mt19937 gen(config.random_seed);
    xt::xarray<float> query_tensor =
        xt::random::rand<float>({config.batch_size, H, config.sequence_length, config.query_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor =
        xt::random::rand<float>({config.batch_size, H, config.sequence_length, config.key_value_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor =
        xt::random::rand<float>({config.batch_size, H, config.sequence_length, config.key_value_dim}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // Convert to device tensors
    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    // Run SDPA kernel
    auto result = ttml::metal::sdpa_fw(
        query,
        key,
        value,
        attn_mask,
        config.num_query_heads,
        config.num_key_heads,
        config.dropout_prob,
        return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

    // Prepare split-head tensors for baseline comparison
    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, config.num_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, config.num_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, config.num_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    // Run baseline implementation
    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor =
        fuse_heads(core::to_xtensor(baseline_result[0]), config.num_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    // Run ground truth float implementation
    auto [expected_result, expected_intermediates] = sdpa_grouped_naive_with_intermediates(
        query_tensor, key_tensor, value_tensor, attn_mask_tensor, config.num_query_heads, config.num_key_heads);

    // Shape validation
    ASSERT_EQ(result_xtensor.shape(), expected_result.shape()) << "Result shape mismatch in " << config.test_name;
    ASSERT_EQ(baseline_result_xtensor.shape(), expected_result.shape())
        << "Baseline shape mismatch in " << config.test_name;
    ASSERT_EQ(interm_xtensor.shape(), expected_intermediates.shape())
        << "Intermediate shape mismatch in " << config.test_name;

    fmt::print("baseline_interm_result shape = {}\n", baseline_interm_xtensor.shape());
    fmt::print("interm_result shape = {}\n", interm_xtensor.shape());
    fmt::print("expected_intermediates shape = {}\n", expected_intermediates.shape());

    // Compute MSE for debugging
    float mse_result = compute_mse(expected_result, result_xtensor);
    float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);
    fmt::print("MSE - Kernel vs Expected: {:.2e}, Baseline vs Expected: {:.2e}\n", mse_result, mse_baseline);

    // Add detailed debug prints for specific tests
    if (config.test_name == "SmallBatch_2H_2KV" || config.test_name == "SingleHead_1H_1KV") {
        fmt::print("\n🔍 DETAILED DEBUG for {}:\n", config.test_name);

        // Check SDPA result element-wise
        fmt::print("=== SDPA Result Detailed Analysis ===\n");
        bool result_passes = xt::allclose(result_xtensor, expected_result, config.result_atol, config.result_rtol);
        fmt::print(
            "SDPA Result allclose: {} (atol={:.2e}, rtol={:.2e})\n",
            result_passes,
            config.result_atol,
            config.result_rtol);

        if (!result_passes) {
            int fail_count = 0;
            std::map<size_t, int> failures_per_row;  // Track failures per sequence position
            const int max_fails_to_show = 20;        // Show more failures

            for (size_t b = 0; b < config.batch_size && fail_count < max_fails_to_show; ++b) {
                for (size_t h = 0; h < 1 && fail_count < max_fails_to_show; ++h) {  // H=1 always
                    for (size_t s = 0; s < config.sequence_length && fail_count < max_fails_to_show; ++s) {
                        for (size_t d = 0; d < config.query_dim && fail_count < max_fails_to_show; ++d) {
                            float kernel_val = result_xtensor(b, h, s, d);
                            float expected_val = expected_result(b, h, s, d);
                            float abs_diff = std::abs(kernel_val - expected_val);
                            float rel_diff = std::abs(abs_diff / expected_val);

                            if (abs_diff > config.result_atol && rel_diff > config.result_rtol) {
                                fmt::print(
                                    "  FAIL[{},{},{},{}]: Kernel={:.6f}, Expected={:.6f}, AbsDiff={:.2e}, "
                                    "RelDiff={:.2e}\n",
                                    b,
                                    h,
                                    s,
                                    d,
                                    kernel_val,
                                    expected_val,
                                    abs_diff,
                                    rel_diff);
                                failures_per_row[s]++;
                                fail_count++;
                            }
                        }
                    }
                }
            }
            if (fail_count >= max_fails_to_show) {
                fmt::print("  ... (showing first {} failures only)\n", max_fails_to_show);
            }

            // Show summary of failures per sequence position
            fmt::print("  📊 Failures per sequence position:\n");
            for (const auto& [seq_pos, count] : failures_per_row) {
                fmt::print("    Seq[{}]: {} failures\n", seq_pos, count);
            }
        }

        // Check intermediate result element-wise
        fmt::print("\n=== Intermediate Result Detailed Analysis ===\n");
        bool interm_passes =
            xt::allclose(interm_xtensor, expected_intermediates, config.intermediate_atol, config.intermediate_rtol);
        fmt::print(
            "Intermediate allclose: {} (atol={:.2e}, rtol={:.2e})\n",
            interm_passes,
            config.intermediate_atol,
            config.intermediate_rtol);

        if (!interm_passes) {
            int fail_count = 0;
            const int max_fails_to_show = 10;

            for (size_t b = 0; b < config.batch_size && fail_count < max_fails_to_show; ++b) {
                for (size_t qh = 0; qh < config.num_query_heads && fail_count < max_fails_to_show; ++qh) {
                    for (size_t s = 0; s < config.sequence_length && fail_count < max_fails_to_show; ++s) {
                        float kernel_val = interm_xtensor(b, qh, s, 0);  // Always 1 in last dim
                        float expected_val = expected_intermediates(b, qh, s, 0);
                        float abs_diff = std::abs(kernel_val - expected_val);
                        float rel_diff = std::abs(abs_diff / expected_val);

                        if (abs_diff > config.intermediate_atol && rel_diff > config.intermediate_rtol) {
                            fmt::print(
                                "  INTERM_FAIL[{},{},{},0]: Kernel={:.6f}, Expected={:.6f}, AbsDiff={:.2e}, "
                                "RelDiff={:.2e}\n",
                                b,
                                qh,
                                s,
                                kernel_val,
                                expected_val,
                                abs_diff,
                                rel_diff);
                            fail_count++;
                        }
                    }
                }
            }
            if (fail_count >= max_fails_to_show) {
                fmt::print("  ... (showing first {} failures only)\n", max_fails_to_show);
            }
        }

        fmt::print("🔍 End detailed debug\n\n");
    }

    // Numerical validation
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, config.result_atol, config.result_rtol))
        << "SDPA result comparison failed in " << config.test_name << " (MSE: " << mse_result << ")";
    EXPECT_TRUE(
        xt::allclose(interm_xtensor, expected_intermediates, config.intermediate_atol, config.intermediate_rtol))
        << "Intermediate result comparison failed in " << config.test_name;

    fmt::print("✅ {} completed successfully\n\n", config.test_name);
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
        .test_name = "SmallBatch_12H_6KV_Production"};
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

    fmt::print("=== Testing Invalid Head Configurations ===\n");

    const uint32_t B = 1U, H = 1U, S = 128U, dQ = 64U, dKV = 64U;
    std::mt19937 gen(42);

    // Create host tensors (not device tensors yet)
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // Test Case 1: Query heads not divisible by key heads (should fail)
    fmt::print("Testing q_heads=5, kv_heads=3 (not divisible)...\n");
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 5U, 3U, 0.0F, false); }, std::exception)
            << "Should fail when q_heads is not divisible by kv_heads";
    }

    // Test Case 2: Zero query heads (should fail)
    fmt::print("Testing q_heads=0...\n");
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 0U, 1U, 0.0F, false); }, std::exception)
            << "Should fail with zero query heads";
    }

    // Test Case 3: Zero key heads (should fail)
    fmt::print("Testing kv_heads=0...\n");
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 0U, 0.0F, false); }, std::exception)
            << "Should fail with zero key heads";
    }

    // Test Case 4: Query heads smaller than key heads (should fail)
    fmt::print("Testing q_heads=2, kv_heads=4 (q < kv)...\n");
    {
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        EXPECT_THROW(
            { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 4U, 0.0F, false); }, std::exception)
            << "Should fail when query heads < key heads";
    }

    fmt::print("✅ Invalid head configuration tests passed\n\n");
}

TEST_F(SDPAForwardTest, ValidationTest_ShapeMismatch) {
    using namespace ttml;

    fmt::print("=== Testing Shape Mismatch Conditions ===\n");

    std::mt19937 gen(42);

    // Test Case 1: Key-Value dimension mismatch
    {
        fmt::print("Testing key-value dimension mismatch...\n");
        xt::xarray<float> query_tensor = xt::random::rand<float>({1, 1, 128, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor = xt::random::rand<float>({1, 1, 128, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> value_tensor =
            xt::random::rand<float>({1, 1, 128, 32}, -1.0F, 1.0F, gen);  // Different dimension
        xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

        {
            auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
            auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
            auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
            auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

            EXPECT_THROW(
                { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 2U, 0.0F, false); },
                std::exception)
                << "Should fail with key-value dimension mismatch";
        }
    }

    // Test Case 2: Batch size mismatch
    {
        fmt::print("Testing batch size mismatch...\n");
        xt::xarray<float> query_tensor = xt::random::rand<float>({2, 1, 128, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor =
            xt::random::rand<float>({1, 1, 128, 64}, -1.0F, 1.0F, gen);  // Different batch size
        xt::xarray<float> value_tensor = xt::random::rand<float>({1, 1, 128, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

        {
            auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
            auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
            auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
            auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

            EXPECT_THROW(
                { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 2U, 0.0F, false); },
                std::exception)
                << "Should fail with batch size mismatch";
        }
    }

    // Test Case 3: Sequence length mismatch
    {
        fmt::print("Testing sequence length mismatch...\n");
        xt::xarray<float> query_tensor = xt::random::rand<float>({1, 1, 128, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> key_tensor = xt::random::rand<float>({1, 1, 64, 64}, -1.0F, 1.0F, gen);  // Different seq len
        xt::xarray<float> value_tensor = xt::random::rand<float>({1, 1, 64, 64}, -1.0F, 1.0F, gen);
        xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

        {
            auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
            auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
            auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
            auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

            EXPECT_THROW(
                { auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 2U, 0.0F, false); },
                std::exception)
                << "Should fail with sequence length mismatch";
        }
    }

    fmt::print("✅ Shape mismatch tests passed\n\n");
}

TEST_F(SDPAForwardTest, ValidationTest_EdgeCaseDimensions) {
    using namespace ttml;

    fmt::print("=== Testing Edge Case Dimensions ===\n");

    std::mt19937 gen(42);

    // Test Case 1: Minimum viable dimensions
    {
        fmt::print("Testing minimum viable dimensions (32x32 tiles)...\n");
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
        fmt::print("Testing single head configuration...\n");
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
        fmt::print("Testing 4:1 grouping ratio...\n");
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
        fmt::print("Testing 8:1 grouping ratio...\n");
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

    fmt::print("✅ Edge case dimension tests passed\n\n");
}

TEST_F(SDPAForwardTest, ValidationTest_IntermediateReturnModes) {
    using namespace ttml;

    fmt::print("=== Testing Intermediate Return Modes ===\n");

    const uint32_t B = 1U, H = 1U, S = 128U, d = 64U;
    std::mt19937 gen(42);

    // Create host tensors (not device tensors yet)
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // Test Case 1: return_intermediates = false
    {
        fmt::print("Testing return_intermediates = false...\n");
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 2U, 0.0F, false);

        EXPECT_TRUE(result[0].has_value()) << "Main result should always be present";
        EXPECT_FALSE(result[1].has_value()) << "Intermediate should be null when return_intermediates=false";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        EXPECT_EQ(result_xtensor.shape(), query_tensor.shape()) << "Result shape should match query shape";
    }

    // Test Case 2: return_intermediates = true
    {
        fmt::print("Testing return_intermediates = true...\n");
        auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
        auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
        auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
        auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());

        auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, 2U, 2U, 0.0F, true);

        EXPECT_TRUE(result[0].has_value()) << "Main result should be present";
        EXPECT_TRUE(result[1].has_value()) << "Intermediate should be present when return_intermediates=true";

        xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
        xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

        EXPECT_EQ(result_xtensor.shape(), query_tensor.shape()) << "Result shape should match query shape";

        // Check intermediate shape: (B, num_query_heads, S, 1)
        std::vector<size_t> expected_interm_shape = {B, 2U, S, 1U};
        EXPECT_EQ(interm_xtensor.shape(), expected_interm_shape) << "Intermediate shape should be (B, q_heads, S, 1)";

        // Verify intermediate values are reasonable (should be positive reciprocals)
        EXPECT_TRUE(xt::all(interm_xtensor > 0.0f)) << "All intermediate values should be positive";
        EXPECT_TRUE(xt::all(interm_xtensor <= 1.0f)) << "All intermediate values should be <= 1.0 (reciprocals)";
    }

    fmt::print("✅ Intermediate return mode tests passed\n\n");
}

TEST_F(SDPAForwardTest, ValidationTest_PerformanceTest) {
    using namespace ttml;

    fmt::print("=== Running Performance Test ===\n");
    uint32_t B = 1U, H = 1U, S = 4096, dQ = 4096U, dKV = 1024;
    uint32_t num_query_heads = 32U;
    uint32_t num_key_heads = 8U;
    float dropout_prob = 0.0F;
    uint32_t random_seed = 42;

    auto* mesh_devices = &autograd::ctx().get_device();
    auto device_ids = mesh_devices->get_device_ids();
    using Clock = std::chrono::high_resolution_clock;

    // Generate test data
    std::mt19937 gen(random_seed);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // Convert to device tensors
    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    // Run SDPA kernel
    auto result = ttml::metal::sdpa_fw(
        query, key, value, attn_mask, num_query_heads, num_key_heads, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor = core::to_xtensor(result[1].value());

    // Prepare split-head tensors for baseline comparison
    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, num_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, num_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, num_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    // Run baseline implementation
    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]), num_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    for (const auto& device_id : device_ids) {
        tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    }
    // Sleep for ten seconds to stabilize GPU frequency
    std::this_thread::sleep_for(std::chrono::seconds(10));

    const int test_size = 50;

    std::vector<ttnn::Tensor> result_holder;
    result_holder.reserve(test_size);

    std::vector<ttnn::Tensor> baseline_holder;
    baseline_holder.reserve(test_size);

    auto op_start = Clock::now();
    for (int i = 0; i < test_size; ++i) {
        result = ttml::metal::sdpa_fw(
            query, key, value, attn_mask, num_query_heads, num_key_heads, dropout_prob, return_intermediates);
        result_holder.push_back(result[0].value());
    }
    // xt::xarray<float> test_result_xtensor = core::to_xtensor(result[0].value());
    // xt::xarray<float> test_interm_xtensor = core::to_xtensor(result[1].value());
    for (const auto& device_id : device_ids) {
        tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    }
    auto op_end = Clock::now();

    // Sleep for ten seconds to stabilize GPU frequency
    std::this_thread::sleep_for(std::chrono::seconds(10));

    auto baseline_start = Clock::now();
    for (int i = 0; i < test_size; ++i) {
        baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
        baseline_holder.push_back(baseline_result[0]);
    }
    // xt::xarray<float> test_baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]),
    // num_query_heads); xt::xarray<float> test_baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);
    for (const auto& device_id : device_ids) {
        tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    }
    auto baseline_end = Clock::now();

    std::chrono::duration<float, std::milli> op_duration = op_end - op_start;
    std::chrono::duration<float, std::milli> baseline_duration = baseline_end - baseline_start;

    std::cout << "op: " << op_duration.count() / static_cast<float>(test_size) << " ms\n";
    std::cout << "baseline : " << baseline_duration.count() / static_cast<float>(test_size) << " ms\n";

    EXPECT_TRUE(false);
}