// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scheduler_base.hpp"

namespace ttml::schedulers {

core::not_null<optimizers::OptimizerBase *> ttml::schedulers::LRSchedulerBase::get_optimizer() const {
    return m_optimizer;
}
LRSchedulerBase::LRSchedulerBase(optimizers::OptimizerBase *optimizer) : m_optimizer(optimizer) {
}

}  // namespace ttml::schedulers
