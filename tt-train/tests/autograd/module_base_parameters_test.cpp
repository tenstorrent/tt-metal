// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <autograd/auto_context.hpp>
#include <memory>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/sgd.hpp"

class Model : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;

public:
    Model() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(784, 128);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);

        create_name("Model");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
    }

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x) {
        x = (*m_fc1)(x);
        x = ttml::ops::relu(x);
        x = (*m_fc2)(x);
        return x;
    }
};

class ModelUnusedLayer : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc3;

public:
    ModelUnusedLayer() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(784, 128);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
        m_fc3 = std::make_shared<ttml::modules::LinearLayer>(64, 32);

        create_name("ModelUnusedLayer");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
        register_module(m_fc3, "fc3");
    }

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x) {
        x = (*m_fc1)(x);
        x = ttml::ops::relu(x);
        x = (*m_fc2)(x);
        return x;
    }
};

class ModuleBaseParametersTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ModuleBaseParametersTest, AllParametersIncluded) {
    Model model;
    auto model_params = model.parameters();
    // 2 LinearLayer modules: 2 weight tensors and 2 bias tensors
    EXPECT_EQ(model_params.size(), 4);
};

TEST_F(ModuleBaseParametersTest, UnusedParametersInModuleSGD) {
    auto* device = &ttml::autograd::ctx().get_device();

    ModelUnusedLayer model;
    auto model_params = model.parameters();
    // 3 LinearLayer modules: 3 weight tensors and 3 bias tensors
    EXPECT_EQ(model_params.size(), 6);
    auto optimizer = ttml::optimizers::SGD(model_params, ttml::optimizers::SGDConfig{});

    auto input_tensor = ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, 1, 1, 784}), device));
    auto output = model(input_tensor);
    output->backward();
    optimizer.step();
}

TEST_F(ModuleBaseParametersTest, UnusedParametersInModuleAdamW) {
    auto* device = &ttml::autograd::ctx().get_device();

    ModelUnusedLayer model;
    auto model_params = model.parameters();
    // 3 LinearLayer modules: 3 weight tensors and 3 bias tensors
    EXPECT_EQ(model_params.size(), 6);
    auto optimizer = ttml::optimizers::AdamW(model_params, ttml::optimizers::AdamWConfig{});

    auto input_tensor = ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, 1, 1, 784}), device));
    auto output = model(input_tensor);
    output->backward();
    optimizer.step();
}
