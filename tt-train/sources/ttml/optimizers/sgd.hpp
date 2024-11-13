// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/optimizer_base.hpp"

namespace ttml::optimizers {

struct SGDConfig {
    float lr{1e-3F};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

class SGD : public OptimizerBase {
public:
    explicit SGD(ttml::autograd::NamedParameters parameters, const SGDConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] autograd::NamedParameters get_state_dict() const override;
    void set_state_dict(const autograd::NamedParameters& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

private:
    size_t steps{0};
    SGDConfig m_config;
    ttml::autograd::NamedParameters m_theta;
};

}  // namespace ttml::optimizers
