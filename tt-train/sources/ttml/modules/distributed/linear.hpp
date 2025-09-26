// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules::distributed {

class RowParallelLinear : public ModuleBase {
public:
    RowParallelLinear(
        uint32_t in_features, uint32_t out_features, bool has_bias = true, bool input_is_parallel = false);
    autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;

private:
    void initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias = true);

    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    bool m_input_is_parallel{false};
};

class ColumnParallelLinear : public ModuleBase {
public:
    ColumnParallelLinear(uint32_t in_features, uint32_t out_features, bool has_bias = true, bool gather_output = false);
    autograd::TensorPtr operator()(const autograd::TensorPtr& tensor) override;

private:
    void initialize_tensors(uint32_t in_features, uint32_t out_features, bool has_bias = true);

    autograd::TensorPtr m_weight;
    autograd::TensorPtr m_bias;
    bool m_gather_output{false};
};

}  // namespace ttml::modules::distributed
