// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "optimizers/adamw.hpp"
#include "tt-metalium/bfloat16.hpp"

class ModelFC : public ttml::modules::ModuleBase {
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

class LanguageModel : public ttml::modules::ModuleBase {
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

namespace {

ttml::autograd::TensorPtr make_host_tensor(const ttnn::Shape& shape) {
    const auto spec = tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::MemoryConfig{}));
    return ttml::autograd::create_tensor(
        ttnn::Tensor::from_vector(std::vector<bfloat16>(shape.volume(), bfloat16{0.0F}), spec));
}

void expect_constructor_error(
    const ttnn::Shape& weight_shape, const ttml::autograd::TensorPtr& bias, const std::string& expected_message) {
    auto weight = make_host_tensor(weight_shape);
    try {
        if (bias == nullptr) {
            (void)ttml::modules::LinearLayer(weight, /* has_bias */ false);
        } else {
            (void)ttml::modules::LinearLayer(weight, bias);
        }
        FAIL() << "LinearLayer accepted unsupported parameter shapes";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected_message), std::string::npos) << error.what();
    }
}

}  // namespace

TEST(LinearLayerConstructorTest, RejectsMalformedInjectedParametersBeforeDeviceInitialization) {
    expect_constructor_error(ttnn::Shape({32}), nullptr, "weight rank 2 through 4");
    expect_constructor_error(ttnn::Shape({1, 1, 1, 64, 32}), nullptr, "weight rank 2 through 4");
    expect_constructor_error(ttnn::Shape({2, 64, 32}), nullptr, "weight to have singleton leading dimensions");
    expect_constructor_error(ttnn::Shape({1, 2, 64, 32}), nullptr, "weight to have singleton leading dimensions");

    expect_constructor_error(
        ttnn::Shape({64, 32}), make_host_tensor(ttnn::Shape({2, 64})), "bias to have singleton leading dimensions");
    expect_constructor_error(
        ttnn::Shape({64, 32}), make_host_tensor(ttnn::Shape({32})), "bias[-1] to match weight[-2]");
}

TEST_F(WeightTyingTest, InfersBiasFromTrailingDimensionsOfCompactInjectedWeights) {
    constexpr uint32_t in_features = 32U;
    constexpr uint32_t out_features = 64U;
    const std::vector<ttnn::Shape> weight_shapes = {
        ttnn::Shape({out_features, in_features}),
        ttnn::Shape({1, out_features, in_features}),
        ttnn::Shape({1, 1, out_features, in_features})};

    for (const auto& weight_shape : weight_shapes) {
        auto weight = ttml::autograd::create_tensor();
        ttml::init::uniform_init(weight, weight_shape, ttml::init::UniformRange{-0.1F, 0.1F});
        auto layer = ttml::modules::LinearLayer(weight, /* has_bias */ true);

        ttml::autograd::TensorPtr bias;
        for (const auto& named_parameter : layer.parameters()) {
            if (named_parameter.second != weight) {
                bias = named_parameter.second;
            }
        }
        ASSERT_NE(bias, nullptr);
        EXPECT_EQ(bias->get_value().logical_shape(), ttnn::Shape({1, 1, 1, out_features}));

        auto input = ttml::autograd::create_tensor();
        ttml::init::uniform_init(input, ttnn::Shape({2, 1, 32, in_features}), ttml::init::UniformRange{-0.1F, 0.1F});
        auto output = layer(input);
        EXPECT_EQ(output->get_value().logical_shape(), ttnn::Shape({2, 1, 32, out_features}));
    }
}

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
    targets.reserve(batch_size * output_features);
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
