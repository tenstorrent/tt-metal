// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/not_null.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {
class OptimizerBase;
}

namespace ttml::schedulers {

class LRSchedulerBase {
public:
    explicit LRSchedulerBase(optimizers::OptimizerBase *optimizer);

    virtual ~LRSchedulerBase() = default;

    virtual void step() = 0;

    [[nodiscard]] virtual float get_last_lr() const = 0;

    [[nodiscard]] virtual float get_current_lr() const = 0;

    [[nodiscard]] core::not_null<optimizers::OptimizerBase *> get_optimizer() const;

    [[nodiscard]] virtual serialization::StateDict get_state_dict() const = 0;
    virtual void set_state_dict(const serialization::StateDict &dict) = 0;

private:
    core::not_null<optimizers::OptimizerBase *> m_optimizer;
};

}  // namespace ttml::schedulers
