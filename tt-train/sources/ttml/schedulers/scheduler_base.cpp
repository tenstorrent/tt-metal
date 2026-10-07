// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scheduler_base.hpp"

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

LRSchedulerBase::LRSchedulerBase(optimizers::OptimizerBase *optimizer) : m_optimizer(optimizer) {
    m_last_lr = m_optimizer->get_initial_lr();
}

core::not_null<optimizers::OptimizerBase *> LRSchedulerBase::get_optimizer() const {
    return m_optimizer;
}

float LRSchedulerBase::get_last_lr() const {
    return m_last_lr;
}

float LRSchedulerBase::get_current_lr() const {
    return m_optimizer->get_lr();
}

void LRSchedulerBase::update_lr(float lr) {
    m_optimizer->set_lr(lr);
    m_last_lr = lr;
}

}  // namespace ttml::schedulers
