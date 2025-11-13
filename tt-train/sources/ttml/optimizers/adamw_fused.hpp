// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

struct AdamWFusedConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{0.0F};
};

class AdamWFused : public OptimizerBase {
public:
    explicit AdamWFused(ttml::serialization::NamedParameters parameters, const AdamWFusedConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override;
    void set_lr(float lr) override;

    [[nodiscard]] float get_beta1() const;
    void set_beta1(float beta1);

    [[nodiscard]] float get_beta2() const;
    void set_beta2(float beta2);

    [[nodiscard]] float get_epsilon() const;
    void set_epsilon(float epsilon);

    [[nodiscard]] float get_weight_decay() const;
    void set_weight_decay(float weight_decay);

private:
    size_t m_steps{0};
    AdamWFusedConfig m_config;
    ttml::serialization::NamedParameters m_first_moment;
    ttml::serialization::NamedParameters m_second_moment;
};

}  // namespace ttml::optimizers
