// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <ttnn/tensor/tensor.hpp>
#include <vector>

#include "optimizers/adamw_utils.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

// Defaults for every param group. Groups can override lr, beta1, beta2, epsilon and weight_decay;
// amsgrad and stochastic_rounding apply to the whole optimizer.
struct AdamWConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{1e-2F};
    bool amsgrad{false};
    // Applies to bf16 parameters only: an fp32 parameter keeps the low bits that rounding would lose.
    bool stochastic_rounding{false};
};

class AdamW : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override;

    explicit AdamW(
        ttml::serialization::NamedParameters parameters,
        const AdamWConfig& config,
        std::vector<ParamGroup> param_groups = {});

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    void set_param_group_option(size_t group_idx, const std::string& key, float value) override;

    [[nodiscard]] bool get_amsgrad() const;
    void set_amsgrad(bool amsgrad);

    [[nodiscard]] bool get_stochastic_rounding() const;
    void set_stochastic_rounding(bool stochastic_rounding);

private:
    void init_max_exp_avg_sq();

    size_t m_steps{0};
    BetaPowers m_beta_pows;
    // Iteration 45 (TTML_ADAMW_DEVICE_STEP=1): per group, a [1, 1, 32, 64] fp32 tensor of [beta1^t | beta2^t] tiles
    // advanced on the device each step, and the step scalars derived from it on the device, so a metal-trace capture
    // of the step replays with advancing powers.
    void reset_beta_pow_tensors();
    ttnn::Tensor advance_step_scalars_on_device(size_t group_idx, float lr);
    bool m_device_step{false};
    std::vector<ttnn::Tensor> m_beta_pow_tensors;
    std::vector<ttnn::Tensor> m_beta_tensors;
    std::vector<ttnn::Tensor> m_tile0_masks;
    std::vector<ttnn::Tensor> m_tile1_masks;
    bool m_amsgrad{false};
    bool m_stochastic_rounding{false};
    ttml::serialization::NamedParameters m_exp_avg;
    ttml::serialization::NamedParameters m_exp_avg_sq;
    ttml::serialization::NamedParameters m_max_exp_avg_sq;
};

}  // namespace ttml::optimizers
