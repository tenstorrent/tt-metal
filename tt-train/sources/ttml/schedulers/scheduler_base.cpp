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

void LRSchedulerBase::step() {
    m_last_step += 1;
    apply_lrs();
}

std::vector<float> LRSchedulerBase::get_last_lrs() const {
    return m_last_lrs;
}

void LRSchedulerBase::apply_lrs() {
    std::vector<float> lrs;
    lrs.reserve(m_base_lrs.size());
    for (const float base_lr : m_base_lrs) {
        lrs.push_back(compute_lr(base_lr));
    }
    m_optimizer->set_group_lrs(lrs);
    m_last_lrs = std::move(lrs);
}

serialization::StateDict LRSchedulerBase::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_base_lrs"] = m_base_lrs;
    res["m_last_lrs"] = m_last_lrs;
    return res;
}

void LRSchedulerBase::validate_state_dict(const serialization::StateDict &dict) const {
    const size_t num_saved = serialization::get_value_type<std::vector<float>>(dict, "m_base_lrs").size();
    const size_t num_groups = m_optimizer->get_group_lrs().size();
    if (num_saved != num_groups) {
        throw std::runtime_error(
            fmt::format("Scheduler checkpoint has {} param groups, but the optimizer has {}", num_saved, num_groups));
    }
}

void LRSchedulerBase::set_state_dict(const serialization::StateDict &dict) {
    validate_state_dict(dict);
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_base_lrs = serialization::get_value_type<std::vector<float>>(dict, "m_base_lrs");
    apply_lrs();
}

}  // namespace ttml::schedulers
