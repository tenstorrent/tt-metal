// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

struct AdamWFullPrecisionConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{1e-2F};
    bool amsgrad{false};
};

class AdamWFullPrecision : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override;

    explicit AdamWFullPrecision(
        ttml::serialization::NamedParameters parameters, const AdamWFullPrecisionConfig& config);

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

    [[nodiscard]] bool get_amsgrad() const;
    void set_amsgrad(bool amsgrad);

    [[nodiscard]] const ttml::serialization::NamedParameters& get_master_weights() const;

private:
    void init_max_exp_avg_sq();

    size_t m_steps{0};
    float m_beta1_pow{1.0F};
    float m_beta2_pow{1.0F};
    AdamWFullPrecisionConfig m_config;
    ttml::serialization::NamedParameters m_master_weights;  // fp32 master weights
    ttml::serialization::NamedParameters m_exp_avg;         // fp32 momentum
    ttml::serialization::NamedParameters m_exp_avg_sq;      // fp32 variance
    ttml::serialization::NamedParameters m_max_exp_avg_sq;  // fp32 for amsgrad
};

}  // namespace ttml::optimizers
