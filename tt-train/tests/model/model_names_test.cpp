// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <autograd/auto_context.hpp>
#include <limits>
#include <memory>
#include <stdexcept>

#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "ops/unary_ops.hpp"

class MNISTModel : public ttml::modules::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc3;
    std::shared_ptr<ttml::modules::DropoutLayer> m_dropout;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm1;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm2;

public:
    MNISTModel() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(784, 128);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
        m_fc3 = std::make_shared<ttml::modules::LinearLayer>(64, 10);
        m_dropout = std::make_shared<ttml::modules::DropoutLayer>(0.2F);

        m_layernorm1 = std::make_shared<ttml::modules::LayerNormLayer>(128);
        m_layernorm2 = std::make_shared<ttml::modules::LayerNormLayer>(10);

        create_name("MNISTModel");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
        register_module(m_fc3, "fc3");
        register_module(m_dropout, "dropout");
        register_module(m_layernorm1, "layernorm1");
        register_module(m_layernorm2, "layernorm2");
    }

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x) {
        x = (*m_dropout)(x);
        x = (*m_fc1)(x);
        x = (*m_layernorm1)(x);
        x = ttml::ops::relu(x);
        x = (*m_fc2)(x);
        x = (*m_layernorm2)(x);
        x = ttml::ops::relu(x);
        x = (*m_fc3)(x);
        return x;
    }
};

class ModelNamesFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST(DropoutLayerTest, RejectsInvalidProbabilityAtConstruction) {
    const std::array<float, 6> invalid_probabilities = {
        -0.1F,
        1.0F,
        1.1F,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (const float probability : invalid_probabilities) {
        EXPECT_THROW(std::make_shared<ttml::modules::DropoutLayer>(probability), std::invalid_argument);
    }

    EXPECT_NO_THROW(std::make_shared<ttml::modules::DropoutLayer>(0.0F));
    EXPECT_NO_THROW(std::make_shared<ttml::modules::DropoutLayer>(0.5F));
}

TEST_F(ModelNamesFullTest, SameModel) {
    MNISTModel model1;
    MNISTModel model2;

    auto model1_params = model1.parameters();
    auto model2_params = model2.parameters();

    EXPECT_EQ(model1_params.size(), model2_params.size());
    for (const auto& [name, tensor] : model1_params) {
        EXPECT_TRUE(model2_params.find(name) != model2_params.end());
    }
};
