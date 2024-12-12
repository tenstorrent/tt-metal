// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

LinearScheduler::LinearScheduler(
    optimizers::OptimizerBase* optimizer, float start_factor, float end_factor, size_t total_steps) :
    LRSchedulerBase(optimizer),
    m_base_lr(optimizer->get_lr()),
    m_last_lr(m_base_lr),
    m_start_factor(start_factor),
    m_end_factor(end_factor),
    m_total_steps(total_steps),
    m_last_step(0) {
}

void LinearScheduler::step() {
    m_last_step += 1;

    float progress = static_cast<float>(m_last_step) / m_total_steps;
    progress = std::min(progress, 1.0f);

    float current_factor = m_start_factor + (m_end_factor - m_start_factor) * progress;
    float new_lr = m_base_lr * current_factor;

    get_optimizer()->set_lr(new_lr);
    m_last_lr = new_lr;
}

void LinearScheduler::set_state_dict(const serialization::StateDict& dict) {
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_last_lr = serialization::get_value_type<float>(dict, "m_last_lr");
}

serialization::StateDict LinearScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    return res;
};

float LinearScheduler::get_last_lr() const {
    return m_last_lr;
}

float LinearScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}

}  // namespace ttml::schedulers
