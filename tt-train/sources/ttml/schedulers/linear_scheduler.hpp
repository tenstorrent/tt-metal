// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/not_null.hpp"
#include "scheduler_base.hpp"

// Assuming necessary includes and that LRSchedulerBase and OptimizerBase are defined
namespace ttml::schedulers {
class LinearScheduler : public LRSchedulerBase {
public:
    LinearScheduler(optimizers::OptimizerBase *optimizer, float end_lr, int total_steps);

    void step() override;

    [[nodiscard]] float get_last_lr() const override;

    [[nodiscard]] float get_current_lr() const override;

private:
    float m_start_lr = 0.F;
    float m_end_lr = 0.F;
    int m_total_steps = 0;
    int m_current_step = 0;
    float m_last_lr = 0.F;
};
}  // namespace ttml::schedulers
