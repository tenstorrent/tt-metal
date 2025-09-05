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

xt::xarray<float> matmul_qk(
    const xt::xarray<float>& Q,  // shape: (B, H, S, d)
    const xt::xarray<float>& K)  // shape: (B, H, S, d)
{
    auto shape = Q.shape();  // (B, H, S, d)
    size_t B = shape[0], H = shape[1], S = shape[2], d = shape[3];

    xt::xarray<float> result = xt::zeros<float>({B, H, S, S});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto q = xt::view(Q, b, h);   // shape: (S, d)
            auto k = xt::view(K, b, h);   // shape: (S, d)
            auto k_t = xt::transpose(k);  // shape: (d, S)
            auto score = dot(q, k_t);     // shape: (S, S)
            xt::view(result, b, h) = score;
        }
    }

    return result;
}

xt::xarray<float> compute_sdpa_without_softmax(
    const xt::xarray<float>& Q,    // shape: (B, H, S, d)
    const xt::xarray<float>& K,    // shape: (B, H, S, d)
    const xt::xarray<float>& V) {  // shape: (B, H, S, d)
    assert(Q.dimension() == 4 && K.dimension() == 4 && V.dimension() == 4);
    assert(Q.shape() == K.shape());
    assert(Q.shape()[0] == V.shape()[0]);  // B
    assert(Q.shape()[1] == V.shape()[1]);  // H
    assert(Q.shape()[2] == V.shape()[2]);  // S_q == S_k == S
    assert(V.shape()[3] == Q.shape()[3]);  // d == d_v

    const size_t B = Q.shape()[0];
    const size_t H = Q.shape()[1];
    const size_t S = Q.shape()[2];
    const size_t d = Q.shape()[3];

    xt::xarray<float> output = xt::zeros<float>({B, H, S, d});

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            auto q = xt::view(Q, b, h);  // shape: (S, d)
            auto k = xt::view(K, b, h);  // shape: (S, d)
            auto v = xt::view(V, b, h);  // shape: (S, d)

            auto k_t = xt::transpose(k);  // shape: (d, S)
            auto qk = dot(q, k_t);        // shape: (S, S)
            // auto qk = dot(q, k);    // shape: (S, S), using K instead of K^T for simplicity
            auto qkv = dot(qk, v);  // shape: (S, d)

            xt::view(output, b, h) = qkv;
        }
    }

    return output;
}

xt::xarray<float> generate_tilewise_symmetric_K(size_t B, size_t H, size_t S, size_t TILE = 32) {
    assert(S % TILE == 0);
    size_t num_tiles = S / TILE;

    xt::xarray<float> K = xt::zeros<float>({B, H, S, S});

    std::mt19937 rng(42);  // reproducible
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t tile_i = 0; tile_i < num_tiles; ++tile_i) {
                for (size_t tile_j = 0; tile_j <= tile_i; ++tile_j) {
                    // Fill tile (i, j)
                    xt::xarray<float> tile = xt::zeros<float>({TILE, TILE});
                    for (size_t i = 0; i < TILE; ++i) {
                        for (size_t j = 0; j < TILE; ++j) {
                            if (tile_i == tile_j && j < i)
                                continue;  // will fill symmetric later

                            float val = dist(rng);
                            tile(i, j) = val;

                            if (tile_i == tile_j)
                                tile(j, i) = val;  // enforce symmetry inside diagonal tile
                        }
                    }

                    // Copy tile to K[b, h]
                    for (size_t i = 0; i < TILE; ++i) {
                        for (size_t j = 0; j < TILE; ++j) {
                            size_t row = tile_i * TILE + i;
                            size_t col = tile_j * TILE + j;
                            K(b, h, row, col) = tile(i, j);
                            if (tile_i != tile_j)
                                K(b, h, col, row) = tile(j, i);  // transpose for symmetric tile
                        }
                    }
                }
            }
        }
    }

    return K;
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

    // test sum exp as intermediate result
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

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 128U, d = 128U;
    // const uint32_t B = 2U, H = 1U, S = 4096U, d = 768U;
    const float dropout_prob = 0.8F;
    const uint32_t num_of_query_heads = 2U;
    const uint32_t num_of_key_heads = 2U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    // xt::xarray<float> attn_mask_tensor = xt::ones<float>({B, H, S, S});
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    auto result = ttml::metal::sdpa_fw(
        query, key, value, attn_mask, num_of_query_heads, num_of_key_heads, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor =
        (return_intermediates) ? core::to_xtensor(result[1].value()) : xt::xarray<float>();

    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, num_of_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, num_of_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, num_of_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]), num_of_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    xt::xarray<float> expected_result = sdpa_grouped_naive(
        query_tensor,
        key_tensor,
        value_tensor,
        attn_mask_tensor,
        /*query_heads=*/num_of_query_heads,
        /*key_heads=*/num_of_key_heads);

    assert((result_xtensor.shape() == expected_result.shape()));
    assert((baseline_result_xtensor.shape() == expected_result.shape()));

    float mse_result = compute_mse(expected_result, result_xtensor);
    float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);

    fmt::print("\n MSE result: {}, baseline MSE: {}\n", mse_result, mse_baseline);

    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < d; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            float baseline_value = baseline_result_xtensor(0, 0, i, j);

            // if (std::abs(actual_value - expected_value) >= 2e-10F + std::abs(expected_value) * 3e-10F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "expected " << expected_value << ", got " << actual_value << ", baseline "
            //               << baseline_value << '\n';
            // }

            // if (std::abs(actual_value - baseline_value) >= 2e-2F + std::abs(baseline_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "baseline " << baseline_value << ", got " << actual_value << '\n';
            // }

            // if (std::abs(baseline_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "baseline " << baseline_value << ", got " << expected_value << '\n';
            // }
        }
    }

    // The equation is: ``std::abs(a - b) <= (m_atol + m_rtol * std::abs(b))``.
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 2e-2F));
    // EXPECT_TRUE(false);

    // EXPECT_TRUE(xt::allclose(interm_xtensor, baseline_interm_xtensor, 3e-2F, 1e-2F));
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch_2Heads_1Group) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 128U, dQ = 128U, dKV = 64U;
    // const uint32_t B = 2U, H = 1U, S = 4096U, d = 768U;
    const float dropout_prob = 0.8F;
    const uint32_t num_of_query_heads = 2U;
    const uint32_t num_of_key_heads = 1U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    // xt::xarray<float> attn_mask_tensor = xt::ones<float>({B, H, S, S});
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    auto result = ttml::metal::sdpa_fw(
        query, key, value, attn_mask, num_of_query_heads, num_of_key_heads, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor =
        (return_intermediates) ? core::to_xtensor(result[1].value()) : xt::xarray<float>();

    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, num_of_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, num_of_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, num_of_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]), num_of_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    xt::xarray<float> expected_result = sdpa_grouped_naive(
        query_tensor,
        key_tensor,
        value_tensor,
        attn_mask_tensor,
        /*query_heads=*/num_of_query_heads,
        /*key_heads=*/num_of_key_heads);

    assert((result_xtensor.shape() == expected_result.shape()));
    assert((baseline_result_xtensor.shape() == expected_result.shape()));

    float mse_result = compute_mse(expected_result, result_xtensor);
    float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);

    fmt::print("\n MSE result: {}, baseline MSE: {}\n", mse_result, mse_baseline);

    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < dQ; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            // float baseline_value = baseline_result_xtensor(0, 0, i, j);

            if (std::abs(actual_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected_value << ", got " << actual_value << '\n';
            }

            // if (std::abs(actual_value - baseline_value) >= 2e-2F + std::abs(baseline_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "baseline " << baseline_value << ", got " << actual_value << '\n';
            // }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 2e-2F));
    // EXPECT_TRUE(xt::allclose(interm_xtensor, baseline_interm_xtensor, 3e-2F, 1e-2F));
    // EXPECT_TRUE(false);
}

TEST_F(SDPAForwardTest, SDPAForwardTest_SmallBatch_12Heads_6Group) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 1024U, dQ = 768U, dKV = 384U;
    // const uint32_t B = 2U, H = 1U, S = 4096U, d = 768U;
    const float dropout_prob = 0.8F;
    const uint32_t num_of_query_heads = 12U;
    const uint32_t num_of_key_heads = 6U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    // xt::xarray<float> attn_mask_tensor = xt::ones<float>({B, H, S, S});
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    auto result = ttml::metal::sdpa_fw(
        query, key, value, attn_mask, num_of_query_heads, num_of_key_heads, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor =
        (return_intermediates) ? core::to_xtensor(result[1].value()) : xt::xarray<float>();

    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, num_of_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, num_of_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, num_of_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]), num_of_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    xt::xarray<float> expected_result = sdpa_grouped_naive(
        query_tensor,
        key_tensor,
        value_tensor,
        attn_mask_tensor,
        /*query_heads=*/num_of_query_heads,
        /*key_heads=*/num_of_key_heads);

    assert((result_xtensor.shape() == expected_result.shape()));
    assert((baseline_result_xtensor.shape() == expected_result.shape()));

    float mse_result = compute_mse(expected_result, result_xtensor);
    float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);

    fmt::print("\n MSE result: {}, baseline MSE: {}\n", mse_result, mse_baseline);

    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < dQ; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            float baseline_value = baseline_result_xtensor(0, 0, i, j);

            if (std::abs(actual_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected_value << ", got " << actual_value << '\n';
            }

            // if (std::abs(actual_value - baseline_value) >= 2e-2F + std::abs(baseline_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "baseline " << baseline_value << ", got " << actual_value << '\n';
            // }

            if (std::abs(baseline_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "baseline " << baseline_value << ", got " << expected_value << '\n';
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 2e-2F));
    // EXPECT_TRUE(false);

    // EXPECT_TRUE(xt::allclose(interm_xtensor, baseline_interm_xtensor, 3e-2F, 1e-2F));
}

TEST_F(SDPAForwardTest, SDPAForwardTest_Batch_12Heads_6Group) {
    using namespace ttml;

    const uint32_t B = 16U, H = 1U, S = 1024U, dQ = 768U, dKV = 384U;
    // const uint32_t B = 2U, H = 1U, S = 4096U, d = 768U;
    const float dropout_prob = 0.8F;
    const uint32_t num_of_query_heads = 12U;
    const uint32_t num_of_key_heads = 6U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, dQ}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, dKV}, -1.0F, 1.0F, gen);
    // xt::xarray<float> attn_mask_tensor = xt::ones<float>({B, H, S, S});
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    auto result = ttml::metal::sdpa_fw(
        query, key, value, attn_mask, num_of_query_heads, num_of_key_heads, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor =
        (return_intermediates) ? core::to_xtensor(result[1].value()) : xt::xarray<float>();

    xt::xarray<float> splited_query_tensor = split_heads(query_tensor, num_of_query_heads);
    xt::xarray<float> splited_key_tensor = split_heads(key_tensor, num_of_key_heads);
    xt::xarray<float> splited_value_tensor = split_heads(value_tensor, num_of_key_heads);

    auto splited_query = core::from_xtensor(splited_query_tensor, &autograd::ctx().get_device());
    auto splited_key = core::from_xtensor(splited_key_tensor, &autograd::ctx().get_device());
    auto splited_value = core::from_xtensor(splited_value_tensor, &autograd::ctx().get_device());

    fmt::print("splited query shape = {}\n", splited_query_tensor.shape());
    fmt::print("splited key shape = {}\n", splited_key_tensor.shape());
    fmt::print("splited value shape = {}\n", splited_value_tensor.shape());

    auto baseline_result = composite_sdpa_fw(splited_query, splited_key, splited_value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = fuse_heads(core::to_xtensor(baseline_result[0]), num_of_query_heads);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    xt::xarray<float> expected_result = sdpa_grouped_naive(
        query_tensor,
        key_tensor,
        value_tensor,
        attn_mask_tensor,
        /*query_heads=*/num_of_query_heads,
        /*key_heads=*/num_of_key_heads);

    assert((result_xtensor.shape() == expected_result.shape()));
    assert((baseline_result_xtensor.shape() == expected_result.shape()));

    float mse_result = compute_mse(expected_result, result_xtensor);
    float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);

    fmt::print("\n MSE result: {}, baseline MSE: {}\n", mse_result, mse_baseline);

    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < dQ; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            float baseline_value = baseline_result_xtensor(0, 0, i, j);

            if (std::abs(actual_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected_value << ", got " << actual_value << '\n';
            }

            // if (std::abs(actual_value - baseline_value) >= 2e-2F + std::abs(baseline_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "baseline " << baseline_value << ", got " << actual_value << '\n';
            // }

            if (std::abs(baseline_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "baseline " << baseline_value << ", got " << expected_value << '\n';
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 2e-2F));
    // EXPECT_TRUE(false);

    // EXPECT_TRUE(xt::allclose(interm_xtensor, baseline_interm_xtensor, 3e-2F, 1e-2F));
}