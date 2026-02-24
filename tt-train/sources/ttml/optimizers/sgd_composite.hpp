// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

struct SGDCompositeConfig {
    float lr{1e-3F};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

class SGDComposite : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "SGDComposite";
    }

    explicit SGDComposite(ttml::serialization::NamedParameters parameters, const SGDCompositeConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override {
        return m_config.lr;
    }

    void set_lr(float lr) override {
        m_config.lr = lr;
    }

private:
    size_t m_steps{0};
    SGDCompositeConfig m_config;
    ttml::serialization::NamedParameters m_theta;
};

}  // namespace ttml::optimizers
