// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

class DropoutLayer : public ModuleBase {
    std::string m_name;
    float m_prob = 0.2F;

public:
    explicit DropoutLayer(float probability);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
