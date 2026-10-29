// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scheduler_base.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

LRSchedulerBase::LRSchedulerBase(optimizers::OptimizerBase *optimizer) :
    m_optimizer(optimizer), m_base_lrs(optimizer->get_initial_group_lrs()), m_last_lrs(m_base_lrs) {
}

core::not_null<optimizers::OptimizerBase *> LRSchedulerBase::get_optimizer() const {
    return m_optimizer;
}

std::vector<float> LRSchedulerBase::get_last_lrs() const {
    return m_last_lrs;
}

serialization::StateDict LRSchedulerBase::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_base_lrs"] = m_base_lrs;
    res["m_last_lrs"] = m_last_lrs;
    return res;
}

void LRSchedulerBase::set_state_dict(const serialization::StateDict &dict) {
    auto last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    auto base_lrs = serialization::get_value_type<std::vector<float>>(dict, "m_base_lrs");
    auto last_lrs = serialization::get_value_type<std::vector<float>>(dict, "m_last_lrs");
    const size_t num_groups = m_optimizer->get_group_lrs().size();
    if (base_lrs.size() != num_groups || last_lrs.size() != num_groups) {
        throw std::runtime_error(fmt::format(
            "Scheduler checkpoint has {} base LRs and {} last LRs, but the optimizer has {} param groups",
            base_lrs.size(),
            last_lrs.size(),
            num_groups));
    }
    restore_own_state(dict);
    m_optimizer->set_group_lrs(last_lrs);
    m_last_step = last_step;
    m_base_lrs = std::move(base_lrs);
    m_last_lrs = std::move(last_lrs);
}

void LRSchedulerBase::restore_own_state(const serialization::StateDict & /*dict*/) {
}

ClosedFormLRScheduler::ClosedFormLRScheduler(optimizers::OptimizerBase *optimizer) : LRSchedulerBase(optimizer) {
}

void ClosedFormLRScheduler::step() {
    m_last_step += 1;
    apply_lrs();
}

void ClosedFormLRScheduler::apply_lrs() {
    std::vector<float> lrs;
    lrs.reserve(m_base_lrs.size());
    for (const float base_lr : m_base_lrs) {
        lrs.push_back(compute_lr(base_lr));
    }
    get_optimizer()->set_group_lrs(lrs);
    m_last_lrs = std::move(lrs);
}

}  // namespace ttml::schedulers
