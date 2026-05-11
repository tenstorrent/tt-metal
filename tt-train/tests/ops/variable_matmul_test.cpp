// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace {

using ttml::metal::VariableMatmulConfig;

float max_abs_error(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    auto a_vec = ttml::core::to_vector<float>(a);
    auto b_vec = ttml::core::to_vector<float>(b);

    float max_err = 0.0F;
    for (size_t i = 0; i < a_vec.size(); ++i) {
        max_err = std::max(max_err, std::abs(a_vec[i] - b_vec[i]));
    }
    return max_err;
}

ttnn::Tensor create_random_device_tensor(uint32_t M, uint32_t K, ttnn::distributed::MeshDevice* device) {
    xt::xarray<float> xt = xt::random::randn<float>({1U, 1U, M, K});
    return ttml::core::from_xtensor(xt, device);
}

const VariableMatmulConfig kConfig{
    .M_block_size = 2,
    .K_block_size = 4,
    .N_block_size = 4,
    .subblock_h = 2,
    .subblock_w = 2,
    .compute_with_storage_grid_size = {10, 10},
};

}  // namespace

class VariableMatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(VariableMatmulTest, Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input, weight, kConfig);
    auto ref = ttnn::matmul(input, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "Max absolute error: " << err;
}

TEST_F(VariableMatmulTest, VariableM_SmallM_32x128x512) {
    const uint32_t M = 32, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input, weight, kConfig);
    auto ref = ttnn::matmul(input, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "Max absolute error: " << err;
}

// Multiple M values — simulates MoE dispatch. At most 2 compilations (transpose variants).
TEST_F(VariableMatmulTest, VariableM_MultipleShapes) {
    const uint32_t K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto weight = create_random_device_tensor(K, N, device);

    for (uint32_t M : {32U, 64U, 128U, 256U, 512U}) {
        auto input = create_random_device_tensor(M, K, device);

        auto result = ttml::metal::variable_matmul(input, weight, kConfig);
        auto ref = ttnn::matmul(input, weight, false, false);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "M=" << M << " max_abs_error: " << err;
    }
}

// transpose_b: weight is stored as [N, K] but interpreted as [K, N] for the matmul.
TEST_F(VariableMatmulTest, TransposeB_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input, weight_nk, cfg);
    auto ref = ttnn::matmul(input, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_b max_abs_error: " << err;
}

TEST_F(VariableMatmulTest, TransposeB_VariableM) {
    const uint32_t K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_b = true;

    for (uint32_t M : {32U, 64U, 128U, 256U, 512U}) {
        auto input = create_random_device_tensor(M, K, device);

        auto result = ttml::metal::variable_matmul(input, weight_nk, cfg);
        auto ref = ttnn::matmul(input, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "transpose_b M=" << M << " max_abs_error: " << err;
    }
}

// transpose_a: input is stored as [K, M] but interpreted as [M, K] for the matmul.
TEST_F(VariableMatmulTest, TransposeA_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);  // stored [K, M]
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a max_abs_error: " << err;
}

// Both transposes simultaneously.
TEST_F(VariableMatmulTest, TransposeBoth_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);   // stored [K, M]
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_a = true;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input_km, weight_nk, cfg);
    auto ref = ttnn::matmul(input_km, weight_nk, /*transpose_a=*/true, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a+b max_abs_error: " << err;
}
