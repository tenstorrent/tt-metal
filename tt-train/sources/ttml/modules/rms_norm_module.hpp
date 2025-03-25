// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/rmsnorm_op.hpp"

namespace ttml::modules {

class RMSNormLayer : public autograd::ModuleBase {
private:
    float m_epsilon = 1e-5F;
    bool m_use_composite = false;
    autograd::TensorPtr m_gamma = nullptr;

public:
    void initialize_tensors(uint32_t features);
    explicit RMSNormLayer(uint32_t features, float epsilon = 1e-5F, bool use_composite = false);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
