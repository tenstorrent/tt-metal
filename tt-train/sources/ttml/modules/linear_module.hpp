// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

class LinearLayer : public ModuleBase {
private:
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    void register_tensors();

public:
    LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias = true);
    LinearLayer(const autograd::TensorPtr& weight, const autograd::TensorPtr& bias);
    LinearLayer(const autograd::TensorPtr& weight, bool has_bias = true);
    autograd::TensorPtr get_weight() const;

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
