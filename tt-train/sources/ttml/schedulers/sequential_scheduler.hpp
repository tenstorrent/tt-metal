// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "scheduler_base.hpp"

namespace ttml::schedulers {

class SequentialScheduler : public LRSchedulerBase {
public:
    // Each element in the schedulers vector is a (scheduler, steps) pair.
    // The scheduler runs for 'steps' times, then we move on to the next one.
    // A little bit different from the PyTorch implementation, where the milestones might be less then the number of
    // schedulers which is misleading
    SequentialScheduler(
        optimizers::OptimizerBase *optimizer,
        std::vector<std::unique_ptr<LRSchedulerBase>> schedulers,
        std::vector<size_t> milestones);

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

protected:
    [[nodiscard]] float compute_lr(float base_lr) const override;

    void apply_lrs() override;

    void restore_own_state(const serialization::StateDict &dict) override;

private:
    std::vector<std::unique_ptr<LRSchedulerBase>> m_schedulers;
    std::vector<size_t> m_milestones;
    // The child that wrote the optimizer's current LRs.
    // step() moves on to the next child only when that child is about to take its first step.
    size_t m_current_scheduler_index = 0;
    // Steps the current child has taken.
    int m_current_step_in_scheduler = 0;
};

}  // namespace ttml::schedulers
