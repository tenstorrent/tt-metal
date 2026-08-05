// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_scheduler.hpp"

#include <tt_stl/assert.hpp>

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

LinearScheduler::LinearScheduler(
    optimizers::OptimizerBase* optimizer, float start_factor, float end_factor, size_t total_steps) :
    LRSchedulerBase(optimizer),
    m_base_lr(optimizer->get_lr()),
    m_start_factor(start_factor),
    m_end_factor(end_factor),
    m_total_steps(total_steps),
    m_last_step(0),
    m_last_lr(m_base_lr) {
    TT_FATAL(total_steps > 0, "total_steps = {} must be greater than zero.", total_steps);
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
    m_base_lr = serialization::get_value_type<float>(dict, "m_base_lr");
    m_start_factor = serialization::get_value_type<float>(dict, "m_start_factor");
    m_end_factor = serialization::get_value_type<float>(dict, "m_end_factor");
    m_total_steps = serialization::get_value_type<int>(dict, "m_total_steps");
}

serialization::StateDict LinearScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    res["m_base_lr"] = m_base_lr;
    res["m_start_factor"] = m_start_factor;
    res["m_end_factor"] = m_end_factor;
    res["m_total_steps"] = m_total_steps;
    return res;
};

float LinearScheduler::get_last_lr() const {
    return m_last_lr;
}

float LinearScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}

}  // namespace ttml::schedulers
