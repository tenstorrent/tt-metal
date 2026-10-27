// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_scheduler.hpp"

#include <tt_stl/assert.hpp>

namespace ttml::schedulers {

StepScheduler::StepScheduler(optimizers::OptimizerBase *optimizer, size_t step_size, float gamma) :
    ClosedFormLRScheduler(optimizer), m_step_size(step_size), m_gamma(gamma) {
    TT_FATAL(step_size > 0, "step_size = {} must be greater than zero.", step_size);
    TT_FATAL(gamma > 0.0f, "gamma = {} must be greater than zero.", gamma);

    // At step 0 the factor is gamma^0 == 1, so every group's LR is (re)set to its base LR.
    apply_lrs();
}

float StepScheduler::compute_lr(float base_lr) const {
    // Every step_size epochs, lr is scaled by gamma
    int num_steps = m_last_step / m_step_size;
    return base_lr * std::pow(m_gamma, static_cast<float>(num_steps));
}

void StepScheduler::restore_own_state(const serialization::StateDict &dict) {
    const auto step_size = serialization::get_value_type<size_t>(dict, "m_step_size");
    const auto gamma = serialization::get_value_type<float>(dict, "m_gamma");
    m_step_size = step_size;
    m_gamma = gamma;
}

serialization::StateDict StepScheduler::get_state_dict() const {
    auto res = ClosedFormLRScheduler::get_state_dict();
    res["m_step_size"] = m_step_size;
    res["m_gamma"] = m_gamma;
    return res;
};

}  // namespace ttml::schedulers
