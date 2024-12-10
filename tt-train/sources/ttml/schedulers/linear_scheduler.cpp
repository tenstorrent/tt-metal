// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {
LinearScheduler::LinearScheduler(optimizers::OptimizerBase *optimizer, float end_lr, int total_steps) :
    LRSchedulerBase(optimizer),
    m_start_lr(optimizer->get_lr()),
    m_end_lr(end_lr),
    m_total_steps(total_steps),
    m_current_step(0),
    m_last_lr(m_start_lr) {
    if (total_steps <= 0) {
        throw std::invalid_argument("total_steps must be a positive integer.");
    }
}
void LinearScheduler::step() {
    m_current_step += 1;

    // Compute progress ratio (clamped at 1.0)
    float progress = static_cast<float>(m_current_step) / m_total_steps;
    progress = std::min(progress, 1.0f);

    // Linearly interpolate between start_lr and end_lr
    float new_lr = m_start_lr + (m_end_lr - m_start_lr) * progress;

    get_optimizer()->set_lr(new_lr);
    m_last_lr = new_lr;
}
float LinearScheduler::get_last_lr() const {
    return m_last_lr;
}
float LinearScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}
}  // namespace ttml::schedulers
