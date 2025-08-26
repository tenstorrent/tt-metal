// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
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

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>  // for xt::zeros

// Naive, correctness-first SDPA with grouped KV (GQA/MQA/MHA).
// Physical layouts:
//   Q: (B, 1, S, qD)
//   K: (B, 1, S, kvD)
//   V: (B, 1, S, kvD)
//   attn_mask: (B, 1, S, S) (additive: 0 for keep, large negative for masked)
// Heads:
//   q_num_heads = H
//   kv_num_heads = G
// Requirement: qD/H == kvD/G  (shared per-head width Dh).
// Returns: (B, 1, S, qD)
xt::xarray<float> sdpa_grouped_naive(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& attn_mask,
    std::uint32_t q_num_heads,
    std::uint32_t kv_num_heads) {
    // ---- shape checks ----
    if (Q.dimension() != 4 || K.dimension() != 4 || V.dimension() != 4) {
        throw std::invalid_argument("Q, K, V must be rank-4 (B,1,S,d).");
    }
    if (attn_mask.dimension() != 4) {
        throw std::invalid_argument("attn_mask must be rank-4 (B,1,S,S).");
    }
    const std::size_t B = Q.shape()[0];
    const std::size_t one = Q.shape()[1];
    const std::size_t S = Q.shape()[2];
    const std::size_t qD = Q.shape()[3];

    if (one != 1)
        throw std::invalid_argument("Q[1] must be 1 (layout B,1,S,d).");
    if (K.shape()[0] != B || K.shape()[1] != 1 || K.shape()[2] != S)
        throw std::invalid_argument("K must match Q in (B,1,S,·).");
    if (V.shape()[0] != B || V.shape()[1] != 1 || V.shape()[2] != S)
        throw std::invalid_argument("V must match Q in (B,1,S,·).");
    if (attn_mask.shape()[0] != B || attn_mask.shape()[1] != 1 || attn_mask.shape()[2] != S ||
        attn_mask.shape()[3] != S)
        throw std::invalid_argument("attn_mask must be (B,1,S,S).");

    const std::size_t kvD = V.shape()[3];
    if (K.shape()[3] != kvD)
        throw std::invalid_argument("K and V last dims must match.");

    const std::size_t H = static_cast<std::size_t>(q_num_heads);
    const std::size_t G = static_cast<std::size_t>(kv_num_heads);
    if (H == 0 || G == 0)
        throw std::invalid_argument("Head counts must be > 0.");
    if (qD % H != 0)
        throw std::invalid_argument("qD must be divisible by q_num_heads.");
    if (kvD % G != 0)
        throw std::invalid_argument("kvD must be divisible by kv_num_heads.");

    const std::size_t Dh_q = qD / H;
    const std::size_t Dh_kv = kvD / G;
    if (Dh_q != Dh_kv)
        throw std::invalid_argument("Per-head dims mismatch: qD/H != kvD/G.");
    const std::size_t Dh = Dh_q;

    // Output: (B,1,S,qD)
    xt::xarray<float> Out = xt::zeros<float>({B, std::size_t(1), S, qD});

    // Helpers
    auto g_of_h = [&](std::size_t h) -> std::size_t {
        // contiguous head blocks, works even if H % G != 0
        return (h * G) / H;
    };
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    // Temporary buffers per row to avoid re-allocations
    std::vector<float> scores_row;
    scores_row.resize(S);

    // ---- main loops ----
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t g = g_of_h(h);
            const std::size_t q_off = h * Dh;   // Q slice offset in last dim
            const std::size_t kv_off = g * Dh;  // K/V slice offset in last dim

            for (std::size_t i = 0; i < S; ++i) {
                // 1) scores_row[j] = (Q[b,0,i,q_off:q_off+Dh] · K[b,0,j,kv_off:kv_off+Dh]) * scale + mask
                float rmax = -INFINITY;
                for (std::size_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    // dot product over Dh
                    for (std::size_t t = 0; t < Dh; ++t) {
                        dot += Q(b, 0, i, q_off + t) * K(b, 0, j, kv_off + t);
                    }
                    float s = dot * scale + attn_mask(b, 0, i, j);
                    scores_row[j] = s;
                    if (s > rmax)
                        rmax = s;
                }

                // 2) softmax normalization for row i
                //    denom = sum(exp(s - rmax))
                float denom = 0.0f;
                for (std::size_t j = 0; j < S; ++j) {
                    denom += std::exp(scores_row[j] - rmax);
                }
                // guard (optional)
                if (denom == 0.0f)
                    denom = 1e-20f;

                // 3) O[b,0,i,q_off : q_off+Dh] = sum_j softmax_ij * V[b,0,j,kv_off:kv_off+Dh]
                //    softmax_ij = exp(s_ij - rmax) / denom
                //    accumulate into a local buffer, then store
                for (std::size_t t = 0; t < Dh; ++t) {
                    float acc = 0.0f;
                    for (std::size_t j = 0; j < S; ++j) {
                        float w = std::exp(scores_row[j] - rmax) / denom;
                        if (w != 0.0f) {
                            acc += w * V(b, 0, j, kv_off + t);
                        }
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

xt::xarray<float> xt_composite_sdpa_fw(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const std::optional<xt::xarray<float>>& attn_mask) {
    xt::xarray<float> result = matmul_qk(query, key);
    return result;
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
    ttnn::Tensor qk_scaled = ttnn_fixed::matmul(q_scaled, key, /*transpose_a=*/false, /*transpose_b=*/true);

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
    auto interm_mm_result = ttnn_fixed::matmul(exp_qk_scaled, value, /*transpose_a=*/false, /*transpose_b=*/false);

    auto attention_weights = ttml::metal::softmax(qk_scaled, /* axis */ 3);

    auto attention_qkv = ttnn_fixed::matmul(attention_weights, value, /*transpose_a=*/false, /*transpose_b=*/false);
    return {attention_qkv, recip_sum_exp};
}

TEST_F(SDPAForwardTest, SDPAForwardTest_MatmulQKV_Small) {
    using namespace ttml;

    const uint32_t B = 2U, H = 1U, S = 4096U, d = 768U;
    const float dropout_prob = 0.8F;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    // xt::xarray<float> attn_mask_tensor = xt::ones<float>({B, H, S, S});
    xt::xarray<float> attn_mask_tensor = generate_mask(query_tensor);

    // for (uint32_t b = 0; b < B; ++b) {
    //     for (uint32_t i = 0; i < S; ++i) {
    //         for (uint32_t j = 0; j < d; ++j) {
    //             query_tensor(b, 0, i, j) = static_cast<float>(1U);
    //             key_tensor(b, 0, i, j) = static_cast<float>(1U);
    //         }
    //     }
    // }
    // std::cout << '\n';
    // for (uint32_t i = 0; i < d; ++i) {
    //     std::cout << key_tensor(0, 0, 64, i) << ' ';
    // }
    // std::cout << '\n';

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(attn_mask_tensor, &autograd::ctx().get_device());
    const bool return_intermediates = true;

    auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, dropout_prob, return_intermediates);
    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    xt::xarray<float> interm_xtensor =
        (return_intermediates) ? core::to_xtensor(result[1].value()) : xt::xarray<float>();

    auto baseline_result = composite_sdpa_fw(query, key, value, attn_mask);
    xt::xarray<float> baseline_result_xtensor = core::to_xtensor(baseline_result[0]);
    xt::xarray<float> baseline_interm_xtensor = core::to_xtensor(baseline_result[1]);

    xt::xarray<float> expected_result = compute_sdpa_without_softmax(query_tensor, key_tensor, value_tensor);

    assert((result_xtensor.shape() == expected_result.shape()));
    assert((baseline_result_xtensor.shape() == expected_result.shape()));

    // float mse_result = compute_mse(expected_result, result_xtensor);
    // float mse_baseline = compute_mse(expected_result, baseline_result_xtensor);

    // fmt::print("\n MSE result: {}, baseline MSE: {}\n", mse_result, mse_baseline);

    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < d; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            float baseline_value = baseline_result_xtensor(0, 0, i, j);

            // if (std::abs(actual_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
            //     std::cout << "Mismatch at (" << i << ", " << j << "): "
            //               << "expected " << expected_value << ", got " << actual_value << ", baseline "
            //               << baseline_result_xtensor(0, 0, i, j) << '\n';
            // }

            if (std::abs(actual_value - baseline_value) >= 2e-2F + std::abs(baseline_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "baseline " << baseline_value << ", got " << actual_value << '\n';
            }
        }
    }
    EXPECT_TRUE(xt::allclose(result_xtensor, baseline_result_xtensor, 3e-2F, 2e-2F));

    if (return_intermediates) {
        assert((interm_xtensor.shape() == baseline_interm_xtensor.shape()));
        for (size_t i = 0; i < S; ++i) {
            for (size_t j = 0; j < 1U /*d*/; ++j) {
                float expected_interm_value = baseline_interm_xtensor(0, 0, i, j);
                float actual_interm_value = interm_xtensor(0, 0, i, j);

                if (std::abs(actual_interm_value - expected_interm_value) >=
                    1e-2F + std::abs(expected_interm_value) * 3e-2F) {
                    std::cout << "Mismatch in intermediate at (" << i << ", " << j << "): "
                              << "expected " << expected_interm_value << ", got " << actual_interm_value << '\n';
                }
            }
        }

        EXPECT_TRUE(xt::allclose(interm_xtensor, baseline_interm_xtensor, 3e-2F, 1e-2F));
    }
}

TEST_F(SDPAForwardTest, SDPAForwardTest_MatmulQK_Batch) {
    using namespace ttml;

    auto* mesh_devices = &autograd::ctx().get_device();
    // const uint32_t B = 64U, H = 1U, S = 1024U, d = 768U;
    const uint32_t B = 8U, H = 1U, S = 768U, d = 768U;
    const auto shape = ttnn::SmallVector<uint32_t>{B, H, S, d};
    const float dropout_prob = 0.8F;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> mask_tensor = xt::ones<float>({B, H, S, S});

    // xt::xarray<float> query_tensor = xt::zeros<float>({B, H, S, d});
    // xt::xarray<float> key_tensor = xt::zeros<float>({B, H, S, d});

    // Fill Q with values row - wise : Q[b][h][i][j] = i * d + j
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < S; ++i) {
            for (size_t j = 0; j < d; ++j) {
                query_tensor(b, 0, i, j) = static_cast<float>(1U);
                key_tensor(b, 0, i, j) = static_cast<float>(1U);  // same as Q for simplicity
            }
        }
    }

    // Fill Q with random values, fill K = K^t
    assert(S == d);  // to make sure K can be transposed easily
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < S; ++i) {
            for (size_t j = 0; j < S; ++j) {
                float random_value = dist(gen);
                query_tensor(b, 0, i, j) = random_value;

                float random_value_2 = dist(gen);
                key_tensor(b, 0, i, j) = random_value_2;  // same as Q for simplicity
                key_tensor(b, 0, j, i) = random_value_2;  // K = K^t
            }
        }
    }

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());
    auto attn_mask = core::from_xtensor(mask_tensor, &autograd::ctx().get_device());

    auto device_ids = mesh_devices->get_device_ids();
    // mesh_devices->disable_and_clear_program_cache();

    const size_t test_count = 100U;

    auto result = ttml::metal::sdpa_fw(query, key, value, attn_mask, dropout_prob, false);
    auto matmul_result = ttnn_fixed::matmul(query, key, false, true);

    xt::xarray<float> expected_result = matmul_qk(query_tensor, key_tensor);
    xt::xarray<float> matmul_result_xtensor = core::to_xtensor(matmul_result);

    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    assert((result_xtensor.shape() == expected_result.shape()));

    for (size_t i = 0; i < 1U; ++i) {
        for (size_t j = 0; j < S; ++j) {
            float expected_value = expected_result(0, 0, i, j);
            float actual_value = result_xtensor(0, 0, i, j);
            float matmul_value = matmul_result_xtensor(0, 0, i, j);

            if (std::abs(actual_value - expected_value) >= 2e-2F + std::abs(expected_value) * 3e-2F) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected_value << ", got " << actual_value << ", matmul " << matmul_value
                          << '\n';
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 2e-2F));
    EXPECT_TRUE(false);
}
