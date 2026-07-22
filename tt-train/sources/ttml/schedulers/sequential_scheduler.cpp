// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sequential_scheduler.hpp"

#include <fmt/format.h>

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace {

// Per-child key prefix used to namespace each wrapped scheduler's state in the
// flat ``StateDict``.
std::string scheduler_prefix(size_t index) {
    return fmt::format("scheduler_{}/", index);
}

}  // namespace

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

    if (m_milestones.size() != m_schedulers.size()) {
        throw std::invalid_argument(fmt::format(
            "SequentialScheduler: milestones.size() ({}) must equal schedulers.size() ({}).",
            m_milestones.size(),
            m_schedulers.size()));
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
    m_current_scheduler_index = serialization::get_value_type<size_t>(dict, "m_current_scheduler_index");

    // Restore every wrapped child scheduler's state. Each child ``i`` reads
    // back its keys from the flat dict under the ``scheduler_{i}/`` prefix.
    //
    // NOTE: This assumes the destination ``SequentialScheduler`` was
    // constructed with the same number of children as the source. A
    // size mismatch will cause the child's ``set_state_dict`` to throw
    // because expected keys won't be present.
    for (size_t i = 0; i < m_schedulers.size(); ++i) {
        const auto prefix = scheduler_prefix(i);
        serialization::StateDict child_dict;
        for (const auto &[key, value] : dict) {
            if (key.compare(0, prefix.size(), prefix) == 0) {
                child_dict[key.substr(prefix.size())] = value;
            }
        }
        m_schedulers[i]->set_state_dict(child_dict);
    }
}
serialization::StateDict SequentialScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_current_step_in_scheduler"] = m_current_step_in_scheduler;
    res["m_last_lr"] = m_last_lr;
    res["m_current_scheduler_index"] = m_current_scheduler_index;

    // Save every wrapped child scheduler's state under the ``scheduler_{i}/``
    // key prefix to avoid name collisions in the flat StateDict layout.
    for (size_t i = 0; i < m_schedulers.size(); ++i) {
        const auto prefix = scheduler_prefix(i);
        for (const auto &[key, value] : m_schedulers[i]->get_state_dict()) {
            res[prefix + key] = value;
        }
    }
    return res;
};

}  // namespace ttml::schedulers
