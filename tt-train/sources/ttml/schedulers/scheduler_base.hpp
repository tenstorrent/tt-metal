// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "core/not_null.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {
class OptimizerBase;
}

namespace ttml::schedulers {

// LRScheduler holds a base learning rate per optimizer param group and on every step computes
// the new learning rate for each group.
class LRSchedulerBase {
public:
    explicit LRSchedulerBase(optimizers::OptimizerBase *optimizer);

    virtual ~LRSchedulerBase() = default;

    // Advances the step count and applies the new LR of every param group.
    virtual void step();

    // LRs this scheduler last computed, one per param group.
    [[nodiscard]] virtual std::vector<float> get_last_lrs() const;

    [[nodiscard]] core::not_null<optimizers::OptimizerBase *> get_optimizer() const;

    // Returns the step count and the per-group base and last LRs.
    // Derived schedulers add their own hyperparameters on top.
    [[nodiscard]] virtual serialization::StateDict get_state_dict() const;

    // Restores the step count, the base LRs and the derived scheduler's own state, then writes the saved last LRs
    // back to the optimizer. Throws without changing the scheduler if the state dict is rejected.
    virtual void set_state_dict(const serialization::StateDict &dict);

private:
    core::not_null<optimizers::OptimizerBase *> m_optimizer;

protected:
    // Reads and assigns the derived scheduler's own state.
    virtual void restore_own_state(const serialization::StateDict &dict);

    // LR of one param group at m_last_step, given the group's base LR.
    [[nodiscard]] virtual float compute_lr(float base_lr) const = 0;

    // Sets every group's LR to compute_lr(its base LR) at m_last_step and records them as the last LRs.
    virtual void apply_lrs();

    // Per-group base LRs, taken from the optimizer's initial LRs so that several schedulers on one optimizer
    // (e.g. a warmup/decay chain) share the same base.
    std::vector<float> m_base_lrs;

    // Per-group LRs this scheduler last wrote. Initialized to the base LRs.
    std::vector<float> m_last_lrs;
    size_t m_last_step = 0;
};

}  // namespace ttml::schedulers
