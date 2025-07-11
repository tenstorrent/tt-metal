// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "optimizers/adamw.hpp"

class ModelFC : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;

public:
    ModelFC() {
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(64, 64);
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(m_fc2->get_weight(), /* has_bias*/ true);
        create_name("ModelFC");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
    }

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x) {
        auto out = (*m_fc1)(x);
        out = ttml::ops::relu(out);
        out = (*m_fc2)(out);
        return out;
    }

    ttml::autograd::TensorPtr get_fc1_weight() {
        return m_fc1->get_weight();
    }

    ttml::autograd::TensorPtr get_fc2_weight() {
        return m_fc2->get_weight();
    }
};

class LanguageModel : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::Embedding> m_emb;

public:
    LanguageModel() {
        m_emb = std::make_shared<ttml::modules::Embedding>(64, 128);
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(m_emb->get_weight(), /* has_bias*/ true);

        create_name("LanguageModel");

        register_module(m_fc1, "fc1");
        register_module(m_emb, "emb");
    }
};

class WeightTyingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(WeightTyingTest, ModelFC) {
    auto model = ModelFC();
    auto params = model.parameters();
    assert(params.size() == 3U);

    std::vector<std::string> names;
    names.reserve(params.size());

    for (const auto& [name, tensor] : params) {
        names.push_back(name);
    }

    std::sort(names.begin(), names.end());
    EXPECT_EQ(names[0], "ModelFC/fc1/bias");
    EXPECT_EQ(names[1], "ModelFC/fc1/weight");
    EXPECT_EQ(names[2], "ModelFC/fc2/bias");

    const size_t batch_size = 64;
    const size_t num_features = 64;
    const size_t output_features = 64;
    std::vector<float> features;
    features.reserve(batch_size * num_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    for (size_t i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_features; ++j) {
            targets.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    auto* device = &ttml::autograd::ctx().get_device();
    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1, 1, num_features}), device));

    auto targets_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, output_features}), device));

    auto optimizer_params = ttml::optimizers::AdamWConfig();
    optimizer_params.lr = 0.01F;
    auto optimizer = ttml::optimizers::AdamW(model.parameters(), optimizer_params);

    for (uint32_t step = 0; step < 5U; ++step) {
        optimizer.zero_grad();
        auto output = model(data_tensor);
        auto loss = ttml::ops::mse_loss(output, targets_tensor);
        loss->backward();
        optimizer.step();
    }

    auto fc1_weight = model.get_fc1_weight();
    auto fc2_weight = model.get_fc2_weight();

    auto fc1_weight_data = ttml::core::to_vector(fc1_weight->get_value());
    auto fc2_weight_data = ttml::core::to_vector(fc2_weight->get_value());

    // check that weights coincide
    EXPECT_EQ(fc1_weight_data.size(), fc2_weight_data.size());
    EXPECT_EQ(fc1_weight_data, fc2_weight_data);
};

TEST_F(WeightTyingTest, LanguageModel) {
    auto model = LanguageModel();
    auto params = model.parameters();
    assert(params.size() == 2U);

    std::vector<std::string> names;
    names.reserve(params.size());
    for (const auto& [name, tensor] : params) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());

    EXPECT_EQ(names[0], "LanguageModel/emb/weight");
    EXPECT_EQ(names[1], "LanguageModel/fc1/bias");
};
