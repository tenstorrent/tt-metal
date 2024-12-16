// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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
    // schedulers which is missleading
    SequentialScheduler(
        optimizers::OptimizerBase *optimizer,
        std::vector<std::unique_ptr<LRSchedulerBase>> schedulers,
        std::vector<size_t> milestones);

    void step() override;

    [[nodiscard]] float get_last_lr() const override;

    [[nodiscard]] float get_current_lr() const override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict &dict) override;

private:
    std::vector<std::unique_ptr<LRSchedulerBase>> m_schedulers;
    std::vector<size_t> m_milestones;
    size_t m_current_scheduler_index = 0;
    int m_current_step_in_scheduler = 0;
    float m_last_lr = 0.F;
};

}  // namespace ttml::schedulers
