// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "models.hpp"

#include <memory>

#include "modules/multi_layer_perceptron.hpp"
#include "ops/unary_ops.hpp"

MNISTModel::MNISTModel() {
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

ttml::autograd::TensorPtr MNISTModel::operator()(ttml::autograd::TensorPtr x) {
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
std::shared_ptr<ttml::modules::MultiLayerPerceptron> create_base_mlp(uint32_t num_features, uint32_t num_targets) {
    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .m_input_features = num_features, .m_hidden_features = {128}, .m_output_features = num_targets};
    return std::make_shared<ttml::modules::MultiLayerPerceptron>(model_params);
}
