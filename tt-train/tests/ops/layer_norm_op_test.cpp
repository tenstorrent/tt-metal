// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <random>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/layernorm_op.hpp"
#include "ops/losses.hpp"

class LayerNormOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(LayerNormOpTest, LayerNormOp_0) {
    using namespace ttml;

    uint32_t batch_size = 6;
    uint32_t seq_len = 13;
    uint32_t heads = 16;
    uint32_t features = 333;

    uint32_t size = batch_size * seq_len * heads;

    std::vector<float> test_data;
    test_data.reserve((size_t)batch_size * seq_len * heads * features);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        float mean = (float)i / (float)size;
        float stddev = 1.F + (float)i / (float)(size * 2);
        std::mt19937 gen(i);
        std::normal_distribution<float> dist(mean, stddev);
        for (uint32_t j = 0; j < features; j++) {
            test_data.push_back(dist(gen));
        }
    }

    auto tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, seq_len, heads, features}), &autograd::ctx().get_device()));

    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    auto result = ops::layernorm(tensor, gamma, beta);

    auto result_tensor = result->get_value();
    auto result_data = core::to_vector(result_tensor);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        uint32_t idx = i * features;

        float exp_mean = 0.F;
        float exp_var = 0.F;
        for (uint32_t j = 0; j < features; ++j) {
            exp_mean += result_data[idx + j];
            exp_var += result_data[idx + j] * result_data[idx + j];
        }

        exp_mean /= (float)features;
        exp_var /= (float)features;
        exp_var = exp_var - exp_mean * exp_mean;

        EXPECT_NEAR(exp_mean, 0.F, 5e-2);
        EXPECT_NEAR(exp_var, 1.F, 5e-2);
    }
}

TEST_F(LayerNormOpTest, LayerNormOp_backward) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 1;
    uint32_t heads = 1;
    uint32_t features = 3;

    std::vector<float> test_data{0.0, 1.0, 2.0};
    auto tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, seq_len, heads, features}), &autograd::ctx().get_device()));

    auto gamma = autograd::create_tensor(
        core::from_vector({1, 2, 3}, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    auto result = ops::layernorm(tensor, gamma, beta);
    auto target = autograd::create_tensor(core::zeros_like(tensor->get_value()));
    result = ops::mse_loss(result, target);
    result->backward();

    auto tensor_grad = core::to_vector(tensor->get_grad());
    auto gamma_grad = core::to_vector(gamma->get_grad());
    auto beta_grad = core::to_vector(beta->get_grad());
    std::vector<float> expected_tensor_grad{1.3333, -2.6667, 1.3333};
    std::vector<float> expected_gamma_grad{1.0000, 0.0000, 3.0000};
    std::vector<float> expected_beta_grad{-0.8165, 0.0000, 2.4495};
    for (uint32_t i = 0; i < features; ++i) {
        EXPECT_NEAR(beta_grad[i], expected_beta_grad[i], 5e-2);
        EXPECT_NEAR(gamma_grad[i], expected_gamma_grad[i], 5e-2);
        EXPECT_NEAR(tensor_grad[i], expected_tensor_grad[i], 6e-2);
    }
}

TEST_F(LayerNormOpTest, CompositeLayerNormOp_0) {
    using namespace ttml;

    uint32_t batch_size = 6;
    uint32_t seq_len = 13;
    uint32_t heads = 16;
    uint32_t features = 333;

    uint32_t size = batch_size * seq_len * heads;

    std::vector<float> test_data;
    test_data.reserve((size_t)batch_size * seq_len * heads * features);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        float mean = (float)i / (float)size;
        float stddev = 1.F + (float)i / (float)(size * 2);
        std::mt19937 gen(i);
        std::normal_distribution<float> dist(mean, stddev);
        for (uint32_t j = 0; j < features; j++) {
            test_data.push_back(dist(gen));
        }
    }

    auto tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, seq_len, heads, features}), &autograd::ctx().get_device()));

    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    auto result = ops::composite_layernorm(tensor, gamma, beta);

    auto result_tensor = result->get_value();
    auto result_data = core::to_vector(result_tensor);
    for (uint32_t i = 0; i < batch_size * seq_len * heads; i++) {
        uint32_t idx = i * features;

        float exp_mean = 0.F;
        float exp_var = 0.F;
        for (uint32_t j = 0; j < features; ++j) {
            exp_mean += result_data[idx + j];
            exp_var += result_data[idx + j] * result_data[idx + j];
        }

        exp_mean /= (float)features;
        exp_var /= (float)features;
        exp_var = exp_var - exp_mean * exp_mean;

        EXPECT_NEAR(exp_mean, 0.F, 3e-2);
        EXPECT_NEAR(exp_var, 1.F, 3e-2);
    }
}

TEST_F(LayerNormOpTest, CompositeLayerNormOp_backward) {
    using namespace ttml;

    uint32_t batch_size = 1;
    uint32_t seq_len = 1;
    uint32_t heads = 1;
    uint32_t features = 3;

    std::vector<float> test_data{0.0, 1.0, 2.0};
    auto tensor = autograd::create_tensor(core::from_vector(
        test_data, ttnn::Shape({batch_size, seq_len, heads, features}), &autograd::ctx().get_device()));

    auto gamma = autograd::create_tensor(
        core::from_vector({1, 2, 3}, ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));
    auto beta = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, 1, features}), &autograd::ctx().get_device()));

    auto result = ops::composite_layernorm(tensor, gamma, beta);
    auto target = autograd::create_tensor(core::zeros_like(tensor->get_value()));
    result = ops::mse_loss(result, target);
    result->backward();

    auto tensor_grad = core::to_vector(tensor->get_grad());
    auto gamma_grad = core::to_vector(gamma->get_grad());
    auto beta_grad = core::to_vector(beta->get_grad());
    std::vector<float> expected_tensor_grad{1.3333, -2.6667, 1.3333};
    std::vector<float> expected_gamma_grad{1.0000, 0.0000, 3.0000};
    std::vector<float> expected_beta_grad{-0.8165, 0.0000, 2.4495};
    for (uint32_t i = 0; i < features; ++i) {
        EXPECT_NEAR(beta_grad[i], expected_beta_grad[i], 3e-2);
        EXPECT_NEAR(gamma_grad[i], expected_gamma_grad[i], 3e-2);
        EXPECT_NEAR(tensor_grad[i], expected_tensor_grad[i], 6e-2);
    }
}
