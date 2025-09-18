// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/no_op.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"

class NoOpFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(NoOpFullTest, NIGHTLY_NoOpTest) {
    using namespace ttml::ops;
    ttml::autograd::ctx().set_seed(123U);
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

    auto optimizer = ttml::optimizers::NoOp(model.parameters());

    const size_t steps = 100U;
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

    EXPECT_EQ(initial_weight_values, current_weight_values);
    EXPECT_EQ(losses.back(), losses.front());
}
