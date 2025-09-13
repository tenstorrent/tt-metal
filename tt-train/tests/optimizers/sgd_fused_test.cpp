// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/sgd_fused.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"

class SGDFusedFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(SGDFusedFullTest, SGDFusedTest) {
    using namespace ttml::ops;
    ttml::autograd::ctx().set_seed(42U);
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32U;
    const size_t num_features = 64U;
    std::vector<float> features;
    features.reserve(batch_size * num_features);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    for (size_t i = 0; i < batch_size; ++i) {
        targets.push_back(static_cast<float>(i) * 0.1F);
    }

    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1U, 1U, num_features}), device));

    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1U, 1U, 1U}), device));

    auto model = ttml::modules::LinearLayer(num_features, 1U);
    auto initial_weight = model.get_weight();
    auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

    auto sgd_fused_config = ttml::optimizers::SGDFusedConfig();
    sgd_fused_config.lr = 1e-4f;
    auto optimizer = ttml::optimizers::SGDFused(model.parameters(), sgd_fused_config);

    const size_t steps = 10U;
    std::vector<float> losses;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        auto loss_value = ttml::core::to_vector(loss->get_value())[0];
        losses.emplace_back(loss_value);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }

    auto current_weight_values = ttml::core::to_vector(model.get_weight()->get_value());

    EXPECT_LT(losses.back(), losses.front());
}

void performTraining(
    std::vector<float>& losses,
    size_t steps,
    ttml::optimizers::OptimizerBase& optimizer,
    ttml::modules::LinearLayer& model,
    ttml::autograd::TensorPtr& data_tensor,
    ttml::autograd::TensorPtr& targets_tensor,
    size_t num_features) {
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        auto loss_value = ttml::core::to_vector(loss->get_value())[0];
        losses.emplace_back(loss_value);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }
}

TEST_F(SGDFusedFullTest, SGDCompositeVsFusedTest) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32U;
    const size_t num_features = 64U;
    std::vector<float> features;
    features.reserve(batch_size * num_features);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    for (size_t i = 0; i < batch_size; ++i) {
        targets.push_back(static_cast<float>(i) * 0.1F);
    }

    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1U, 1U, num_features}), device));

    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1U, 1U, 1U}), device));

    constexpr float learning_rate = 1e-3f;
    const size_t steps = 100U;

    std::vector<float> fusedLosses;
    fusedLosses.reserve(steps);
    ttml::autograd::ctx().set_seed(42U);
    {
        auto model = ttml::modules::LinearLayer(num_features, 1U);
        auto initial_weight = model.get_weight();
        auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

        auto config = ttml::optimizers::SGDFusedConfig();
        config.lr = learning_rate;

        auto fusedOptimizer = ttml::optimizers::SGDFused(model.parameters(), config);
        performTraining(fusedLosses, steps, fusedOptimizer, model, data_tensor, targets_tensor, num_features);
    }

    std::vector<float> compositeLosses;
    compositeLosses.reserve(steps);
    ttml::autograd::ctx().set_seed(42U);
    {
        auto model = ttml::modules::LinearLayer(num_features, 1U);
        auto initial_weight = model.get_weight();
        auto initial_weight_values = ttml::core::to_vector(initial_weight->get_value());

        auto config = ttml::optimizers::SGDConfig();
        config.lr = learning_rate;

        auto compositeOptimizer = ttml::optimizers::SGD(model.parameters(), config);
        performTraining(compositeLosses, steps, compositeOptimizer, model, data_tensor, targets_tensor, num_features);
    }

    constexpr float tol = 1e-3f;
    EXPECT_NEAR(fusedLosses.back(), compositeLosses.back(), tol);
}
