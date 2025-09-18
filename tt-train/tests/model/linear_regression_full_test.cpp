// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"

class LinearRegressionFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(LinearRegressionFullTest, TestLinearRegressionFull) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 128;
    const size_t num_features = 64;
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
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1, 1, num_features}), device));

    auto targets_tensor =
        ttml::autograd::create_tensor(ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, 1}), device));

    auto model = ttml::modules::LinearLayer(num_features, 1);
    auto optimizer = ttml::optimizers::SGD(model.parameters(), {0.01F, 0.0F});

    const size_t steps = 10;
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }
}
