// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "scheduler_base.hpp"

namespace ttml::schedulers {

// Runs other schedulers one after another, each for its milestone's number of steps.
class SequentialScheduler : public LRSchedulerBase {
public:
    // schedulers[i] runs for milestones[i] steps, then the next one takes over.
    SequentialScheduler(
        optimizers::OptimizerBase *optimizer,
        std::vector<std::unique_ptr<LRSchedulerBase>> schedulers,
        std::vector<size_t> milestones);

    // Steps the active scheduler, moving to the next one when its steps are used up.
    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

protected:
    void restore_own_state(const serialization::StateDict &dict) override;

private:
    void apply_lrs();

    std::vector<std::unique_ptr<LRSchedulerBase>> m_schedulers;
    std::vector<size_t> m_milestones;
    // The child that wrote the optimizer's current LRs.
    // step() moves on to the next child only when that child is about to take its first step.
    size_t m_current_scheduler_index = 0;
    // Steps the current child has taken.
    int m_current_step_in_scheduler = 0;
};

}  // namespace ttml::schedulers
