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
    m_base_lr(optimizer->get_initial_lr()),
    m_start_factor(start_factor),
    m_end_factor(end_factor),
    m_total_steps(total_steps),
    m_last_step(0),
    m_last_lr(m_base_lr) {
    TT_FATAL(total_steps > 0, "total_steps = {} must be greater than zero.", total_steps);
    // Bounds match PyTorch's LinearLR. start_factor == 0 is rejected because
    // the factor is applied at construction, so it would make the first
    // optimizer step a no-op (LR = 0).
    TT_FATAL(
        start_factor > 0.0F && start_factor <= 1.0F,
        "start_factor = {} expected to be greater than 0 and less or equal to 1.",
        start_factor);
    TT_FATAL(end_factor >= 0.0F && end_factor <= 1.0F, "end_factor = {} expected to be between 0 and 1.", end_factor);

    // Mirror PyTorch, which applies start_factor at construction: the LR used
    // before the first step() is already base_lr * start_factor.
    m_last_lr = m_base_lr * m_start_factor;
    optimizer->set_lr(m_last_lr);
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
    // Push the restored live LR back to the optimizer: the constructor wrote
    // the construction-time LR, so if the optimizer's state was loaded before
    // the scheduler was constructed, the checkpoint's live LR was overwritten
    // and the first resumed step would otherwise run at the wrong LR.
    get_optimizer()->set_lr(m_last_lr);
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
