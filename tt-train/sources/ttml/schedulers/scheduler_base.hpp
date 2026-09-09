// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

    // Last LR this scheduler computed (recorded by update_lr).
    // SequentialScheduler overrides this to delegate to its active child.
    [[nodiscard]] virtual float get_last_lr() const;

    // Live LR currently held by the optimizer.
    [[nodiscard]] virtual float get_current_lr() const;

    [[nodiscard]] core::not_null<optimizers::OptimizerBase *> get_optimizer() const;

    [[nodiscard]] virtual serialization::StateDict get_state_dict() const = 0;
    virtual void set_state_dict(const serialization::StateDict &dict) = 0;

private:
    core::not_null<optimizers::OptimizerBase *> m_optimizer;

protected:
    // Write ``lr`` to the optimizer and record it as this scheduler's last LR.
    // C++ counterpart of the Python ``_SchedulerBase._apply_initial_lr`` mixin,
    // shared by every site that publishes a new LR:
    //   * constructors — mirror PyTorch's construction-time initial step: each
    //     scheduler applies its step-0 LR at construction;
    //   * step() — publish the newly computed LR;
    //   * set_state_dict() — push the restored live LR back to the optimizer.
    //     The constructor wrote the construction-time LR, so if the optimizer's
    //     state was loaded before the scheduler was constructed, the
    //     checkpoint's live LR was overwritten and the first resumed step
    //     would otherwise run at the wrong LR.
    void update_lr(float lr);

    // Last LR recorded for this scheduler. Initialized to the optimizer's
    // initial (base) LR, mirroring the Python ``_SchedulerBase``. Derived
    // classes read it for state dicts; SequentialScheduler::step also writes
    // it directly because the active child has already written the optimizer.
    float m_last_lr = 0.F;
};

}  // namespace ttml::schedulers
