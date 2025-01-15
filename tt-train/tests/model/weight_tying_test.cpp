// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/module_base.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "ops/unary_ops.hpp"

class ModelFC : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;

public:
    ModelFC() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
        create_name("ModelFC");

        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");

        m_fc1->set_weight(m_fc2->get_weight());
    }
};

class LanguageModel : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::Embedding> m_emb;

public:
    LanguageModel() {
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(128, 64);
        m_emb = std::make_shared<ttml::modules::Embedding>(64, 128);
        create_name("LanguageModel");

        register_module(m_fc1, "fc1");
        register_module(m_emb, "emb");

        m_fc1->set_weight(m_emb->get_weight());
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
