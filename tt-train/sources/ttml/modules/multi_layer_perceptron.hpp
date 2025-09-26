// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

struct MultiLayerPerceptronParameters {
    uint32_t input_features{};
    std::vector<uint32_t> hidden_features;
    uint32_t output_features{};
};

class MultiLayerPerceptron : public ModuleBase {
private:
    std::vector<std::shared_ptr<LinearLayer>> m_layers;

public:
    explicit MultiLayerPerceptron(const MultiLayerPerceptronParameters& params);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
