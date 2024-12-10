// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {
SequentialScheduler::SequentialScheduler(
    optimizers::OptimizerBase *optimizer,
    std::vector<std::unique_ptr<LRSchedulerBase>> schedulers,
    std::vector<size_t> milestones) :
    LRSchedulerBase(optimizer),
    m_schedulers(std::move(schedulers)),
    m_milestones(std::move(milestones)),
    m_current_scheduler_index(0),
    m_current_step_in_scheduler(0),
    m_last_lr(optimizer->get_lr()) {
    if (m_schedulers.empty()) {
        throw std::invalid_argument("SequentialScheduler requires at least one scheduler.");
    }

    // Validate that each scheduler is non-null
    for (auto &scheduler : m_schedulers) {
        if (!scheduler) {
            throw std::invalid_argument("Null scheduler provided to SequentialScheduler.");
        }
    }
}
void SequentialScheduler::step() {
    if (m_current_scheduler_index >= m_schedulers.size()) {
        return;
    }

    auto &current_scheduler = m_schedulers[m_current_scheduler_index];
    auto current_sched_steps = m_milestones[m_current_scheduler_index];
    current_scheduler->step();
    m_current_step_in_scheduler += 1;
    m_last_lr = current_scheduler->get_last_lr();

    if (m_current_step_in_scheduler >= current_sched_steps) {
        m_current_scheduler_index += 1;
        m_current_step_in_scheduler = 0;
    }
}
float SequentialScheduler::get_last_lr() const {
    if (m_current_scheduler_index == 0) {
        return (m_current_scheduler_index < m_schedulers.size())
                   ? m_schedulers[m_current_scheduler_index]->get_last_lr()
                   : m_last_lr;
    } else if (m_current_scheduler_index < m_schedulers.size()) {
        return m_schedulers[m_current_scheduler_index]->get_last_lr();
    }
    return m_last_lr;
}
float SequentialScheduler::get_current_lr() const {
    // The current LR of the optimizer should reflect the last scheduler's step
    return get_optimizer()->get_lr();
}

void SequentialScheduler::set_state_dict(const serialization::StateDict &dict) {
    m_current_step_in_scheduler = serialization::get_value_type<int>(dict, "m_current_step_in_scheduler");
    m_last_lr = serialization::get_value_type<float>(dict, "m_last_lr");
}
serialization::StateDict SequentialScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_current_step_in_scheduler"] = m_current_step_in_scheduler;
    res["m_last_lr"] = m_last_lr;
    return res;
};

}  // namespace ttml::schedulers
