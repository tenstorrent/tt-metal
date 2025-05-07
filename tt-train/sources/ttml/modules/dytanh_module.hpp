// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

class DyTanhLayer : public autograd::ModuleBase {
public:
    std::shared_ptr<autograd::Tensor> m_gain{};
    std::shared_ptr<autograd::Tensor> m_bias{};
    std::shared_ptr<autograd::Tensor> m_scale{};

public:
    void initialize_tensors(uint32_t features, float scale);
    explicit DyTanhLayer(uint32_t features, float scale = 0.5F);  // α = 0.5F following Zhu et al.

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
