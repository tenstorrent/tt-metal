// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "scheduler_base.hpp"

namespace ttml::schedulers {

class StepScheduler : public LRSchedulerBase {
public:
    StepScheduler(optimizers::OptimizerBase *optimizer, size_t step_size, float gamma = 0.1f);

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

    void set_state_dict(const serialization::StateDict &dict) override;

private:
    size_t m_step_size = 0;
    float m_gamma = 0;
    size_t m_last_step = 0;

    float m_base_lr = 0.F;
};

}  // namespace ttml::schedulers
