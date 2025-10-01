// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

class DropoutLayer : public ModuleBase {
    std::string m_name;
    float m_prob = 0.2F;
    bool m_use_per_device_seed{};

public:
    explicit DropoutLayer(float probability, bool use_per_device_seed = true);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
