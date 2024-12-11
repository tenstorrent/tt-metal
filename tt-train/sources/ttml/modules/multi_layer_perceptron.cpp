// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_layer_perceptron.hpp"

#include "modules/linear_module.hpp"

namespace ttml::modules {

template <typename Layers, typename... Args>
void add_linear_layer(Layers& layers, Args&&... args) {
    layers.push_back(std::make_shared<LinearLayer>(std::forward<Args>(args)...));
}

MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptronParameters& params) {
    uint32_t current_input_features = params.input_features;
    for (auto hidden_features : params.hidden_features) {
        add_linear_layer(m_layers, current_input_features, hidden_features);
        current_input_features = hidden_features;
    }
    add_linear_layer(m_layers, current_input_features, params.output_features);

    create_name("mlp");

    for (size_t idx = 0; idx < m_layers.size(); ++idx) {
        register_module(m_layers[idx], "layer_" + std::to_string(idx));
    }
}
autograd::TensorPtr MultiLayerPerceptron::operator()(autograd::TensorPtr tensor) {
    for (size_t index = 0; index < m_layers.size(); ++index) {
        tensor = (*m_layers[index])(tensor);
        if (index + 1 != m_layers.size()) {
            tensor = ops::relu(tensor);
        }
    }

    return tensor;
}

}  // namespace ttml::modules
