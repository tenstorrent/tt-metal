// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>

#include "optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

struct AdamWConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{0.01F};
    // TODO: add amsgrad

    // flag to enable kahan summation to reduce floating point errors
    bool use_kahan_summation{false};
};

class MorehAdamW : public OptimizerBase {
public:
    MorehAdamW(serialization::NamedParameters parameters, const AdamWConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override;
    void set_lr(float lr) override;

private:
    size_t m_steps{0};
    AdamWConfig m_config;
    serialization::NamedParameters m_first_moment;
    serialization::NamedParameters m_second_moment;
};

class AdamW : public OptimizerBase {
public:
    AdamW(serialization::NamedParameters parameters, const AdamWConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override;

    void set_lr(float lr) override;

private:
    size_t m_steps{0};
    AdamWConfig m_config;
    serialization::NamedParameters m_first_moment;
    serialization::NamedParameters m_second_moment;
    serialization::NamedParameters m_kahan_compensation;
};

}  // namespace ttml::optimizers
