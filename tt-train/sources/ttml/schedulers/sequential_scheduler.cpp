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
    LRSchedulerBase(optimizer), m_schedulers(std::move(schedulers)), m_milestones(std::move(milestones)) {
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

    // The children were constructed back-to-back on the same optimizer, so the
    // optimizer currently holds the LAST child's construction-time LRs. Only the
    // first child is active; redo its initial step. Mirrors PyTorch's
    // SequentialLR, which redoes the initial step of the first scheduler only.
    apply_lrs();
}

void SequentialScheduler::apply_lrs() {
    m_last_lrs = m_schedulers[m_current_scheduler_index]->get_last_lrs();
    get_optimizer()->set_group_lrs(m_last_lrs);
}

void SequentialScheduler::step() {
    if (static_cast<size_t>(m_current_step_in_scheduler) >= m_milestones[m_current_scheduler_index]) {
        if (m_current_scheduler_index + 1 >= m_schedulers.size()) {
            return;  // every child has run all its steps
        }
        m_current_scheduler_index += 1;
        m_current_step_in_scheduler = 0;
    }

    auto &scheduler = *m_schedulers[m_current_scheduler_index];
    scheduler.step();
    m_current_step_in_scheduler += 1;
    m_last_lrs = scheduler.get_last_lrs();
}

void SequentialScheduler::restore_own_state(const serialization::StateDict &dict) {
    const auto index = serialization::get_value_type<size_t>(dict, "m_current_scheduler_index");
    const auto step_in_scheduler = serialization::get_value_type<int>(dict, "m_current_step_in_scheduler");
    if (index >= m_schedulers.size()) {
        throw std::invalid_argument(fmt::format(
            "SequentialScheduler state has scheduler index {}, but the chain has {} schedulers",
            index,
            m_schedulers.size()));
    }
    if (step_in_scheduler < 0 || static_cast<size_t>(step_in_scheduler) > m_milestones[index]) {
        throw std::invalid_argument(fmt::format(
            "SequentialScheduler state has {} steps in scheduler {}, outside [0, {}]",
            step_in_scheduler,
            index,
            m_milestones[index]));
    }

    // Restore every wrapped child scheduler's state. Each child ``i`` reads
    // back its keys from the flat dict under the ``scheduler_{i}/`` prefix.
    //
    // NOTE: This assumes the destination ``SequentialScheduler`` was
    // constructed with the same number of children as the source. A
    // size mismatch will cause the child's ``set_state_dict`` to throw
    // because expected keys won't be present.
    //
    // A later child can still reject its state after earlier ones were restored (and wrote their LRs to the
    // optimizer), so snapshot everything and roll back on failure.
    std::vector<serialization::StateDict> snapshots;
    snapshots.reserve(m_schedulers.size());
    for (const auto &scheduler : m_schedulers) {
        snapshots.push_back(scheduler->get_state_dict());
    }
    const auto live_lrs = get_optimizer()->get_group_lrs();
    try {
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
    } catch (...) {
        for (size_t i = 0; i < m_schedulers.size(); ++i) {
            m_schedulers[i]->set_state_dict(snapshots[i]);
        }
        get_optimizer()->set_group_lrs(live_lrs);
        throw;
    }
    // Only after the children accepted their states, so a rejected state dict leaves the chain's position unchanged.
    m_current_scheduler_index = index;
    m_current_step_in_scheduler = step_in_scheduler;
}
serialization::StateDict SequentialScheduler::get_state_dict() const {
    auto res = LRSchedulerBase::get_state_dict();
    res["m_current_step_in_scheduler"] = m_current_step_in_scheduler;
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
