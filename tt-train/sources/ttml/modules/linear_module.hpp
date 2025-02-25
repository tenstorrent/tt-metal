// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

class LinearLayer : public autograd::ModuleBase {
private:
    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;

    void initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias = true);

public:
    LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias = true);

    autograd::TensorPtr get_weight() const;
    void set_weight(const autograd::TensorPtr& weight);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
