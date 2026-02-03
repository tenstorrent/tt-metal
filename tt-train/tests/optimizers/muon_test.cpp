// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/muon.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "fmt/base.h"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"

class MuonFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(MuonFullTest, NIGHTLY_MuonTest) {
    using namespace ttml::ops;
    ttml::autograd::ctx().set_seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32;
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
    auto muon_config = ttml::optimizers::MuonConfig();
    muon_config.lr = 1e-2F;
    muon_config.momentum = 0.95F;
    muon_config.ns_steps = 5;
    auto optimizer = ttml::optimizers::Muon(model.parameters(), muon_config);

    const size_t steps = 100;
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
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_LT(losses.back(), 1e-3F);
}
