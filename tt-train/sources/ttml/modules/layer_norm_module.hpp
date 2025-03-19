// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "ops/layernorm_op.hpp"

namespace ttml::modules {

class LayerNormLayer : public autograd::ModuleBase {
private:
    bool m_use_composite_op = false;
    autograd::TensorPtr m_gamma;
    autograd::TensorPtr m_beta;

public:
    void initialize_tensors(uint32_t features);
    explicit LayerNormLayer(uint32_t features, bool use_composite_op = false);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;
};

}  // namespace ttml::modules
