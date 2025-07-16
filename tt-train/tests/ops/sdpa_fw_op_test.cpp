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

xt::xarray<float> xt_composite_sdpa_fw(
    const xt::xarray<float>& query,
    const xt::xarray<float>& key,
    const xt::xarray<float>& value,
    const std::optional<xt::xarray<float>>& attn_mask) {
    xt::xarray<float> result = matmul_qk(query, key);
    return result;
}

TEST_F(SDPAForwardTest, SDPAForwardTest_MatmulQK_Small) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 32U, d = 32U;
    const auto shape = ttnn::SmallVector<uint32_t>{B, H, S, d};
    const float dropout_prob = 0.8F;

    std::random_device rd;
    std::mt19937 gen(42);
    // xt::xarray<float> query_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    // xt::xarray<float> key_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);
    xt::xarray<float> value_tensor = xt::random::rand<float>({B, H, S, d}, -1.0F, 1.0F, gen);

    xt::xarray<float> query_tensor = xt::zeros<float>({B, H, S, d});
    xt::xarray<float> key_tensor = xt::zeros<float>({B, H, S, d});

    // Fill Q with values row-wise: Q[b][h][i][j] = i * d + j
    for (size_t i = 0; i < S; ++i) {
        for (size_t j = 0; j < d; ++j) {
            query_tensor(0, 0, i, j) = static_cast<float>(i);
            key_tensor(0, 0, i, j) = static_cast<float>(i);  // same as Q for simplicity
        }
    }

    auto query = core::from_xtensor(query_tensor, &autograd::ctx().get_device());
    auto key = core::from_xtensor(key_tensor, &autograd::ctx().get_device());
    auto value = core::from_xtensor(value_tensor, &autograd::ctx().get_device());

    auto result = ttml::metal::sdpa_fw(query, key, value, std::nullopt, dropout_prob, false);

    xt::xarray<float> expected_result = matmul_qk(query_tensor, key_tensor);

    xt::xarray<float> result_xtensor = core::to_xtensor(result[0].value());
    assert((result_xtensor.shape() == expected_result.shape()));

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
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
    using Clock = std::chrono::high_resolution_clock;

    // auto temp_res = ttml::metal::sdpa_fw(query, key, value, attn_mask, dropout_prob, false);
    // auto baseline_temp_res = ttnn_fixed::matmul(query, key, false, true);

    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }

    // auto op_start = Clock::now();
    // for (size_t i = 0; i < test_count; ++i) {
    //     temp_res = ttml::metal::sdpa_fw(query, key, value, std::nullopt, dropout_prob, false);
    // }
    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }
    // // xt::xarray<float> temp_res_xtensor = core::to_xtensor(temp_res[0].value());
    // auto op_end = Clock::now();

    // auto ttnn_matmul_start = Clock::now();
    // for (size_t i = 0; i < test_count; ++i) {
    //     baseline_temp_res = ttnn_fixed::matmul(query, key, false, true);
    //     // xt::xarray<float> baseline_temp_res_xtensor = core::to_xtensor(baseline_temp_res);
    // }
    // for (const auto& device_id : device_ids) {
    //     tt::tt_metal::Synchronize(mesh_devices->get_device(device_id));
    // }
    // // xt::xarray<float> baseline_temp_res_xtensor = core::to_xtensor(baseline_temp_res);
    // auto ttnn_matmul_end = Clock::now();

    // std::chrono::duration<float, std::milli> duration = op_end - op_start;
    // std::chrono::duration<float, std::milli> ttnn_duration = ttnn_matmul_end - ttnn_matmul_start;

    // std::cout << "Our MatMul: " << duration.count() << " ms\n";
    // std::cout << "Reference : " << ttnn_duration.count() << " ms\n";

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
