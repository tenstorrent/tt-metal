// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "autograd/module_base.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_layer_perceptron.hpp"

class MNISTModel : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc3;
    std::shared_ptr<ttml::modules::DropoutLayer> m_dropout;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm1;
    std::shared_ptr<ttml::modules::LayerNormLayer> m_layernorm2;

public:
    MNISTModel();

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x);
};

std::shared_ptr<ttml::modules::MultiLayerPerceptron> create_base_mlp(uint32_t num_features, uint32_t num_targets);
