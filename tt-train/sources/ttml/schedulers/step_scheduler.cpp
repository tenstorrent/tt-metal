// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "step_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

StepScheduler::StepScheduler(optimizers::OptimizerBase *optimizer, size_t step_size, float gamma) :
    LRSchedulerBase(optimizer),
    m_step_size(step_size),
    m_gamma(gamma),
    m_last_step(0),
    m_base_lr(optimizer->get_lr()),
    m_last_lr(m_base_lr) {
    if (gamma <= 0.0f) {
        throw std::invalid_argument(fmt::format("gamma = {} must be greater than zero.", gamma));
    }
}
void StepScheduler::step() {
    m_last_step += 1;

    // Every step_size epochs, lr is scaled by gamma
    int num_steps = m_last_step / m_step_size;
    float new_lr = m_base_lr * std::pow(m_gamma, static_cast<float>(num_steps));

    get_optimizer()->set_lr(new_lr);
    m_last_lr = new_lr;
}
float StepScheduler::get_last_lr() const {
    return m_last_lr;
}
float StepScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}

void StepScheduler::set_state_dict(const serialization::StateDict &dict) {
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_last_lr = serialization::get_value_type<float>(dict, "m_last_lr");
}
serialization::StateDict StepScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    return res;
};

}  // namespace ttml::schedulers
