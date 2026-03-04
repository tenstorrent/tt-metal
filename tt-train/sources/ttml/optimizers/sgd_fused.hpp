// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

struct SGDFusedConfig {
    float lr{1e-3F};
    float momentum{0.0F};
    float dampening{0.0F};
    float weight_decay{0.0F};
    bool nesterov{false};
};

class SGDFused : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "SGDFused";
    }

    explicit SGDFused(ttml::serialization::NamedParameters parameters, const SGDFusedConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override;

    void set_lr(float lr) override;

    [[nodiscard]] float get_momentum() const;
    void set_momentum(float momentum);

    [[nodiscard]] float get_dampening() const;
    void set_dampening(float dampening);

    [[nodiscard]] float get_weight_decay() const;
    void set_weight_decay(float weight_decay);

    [[nodiscard]] bool get_nesterov() const;
    void set_nesterov(bool nesterov);

private:
    void validate_config() const;
    size_t m_steps{0};
    SGDFusedConfig m_config;
    ttml::serialization::NamedParameters m_momentum;
};

}  // namespace ttml::optimizers
