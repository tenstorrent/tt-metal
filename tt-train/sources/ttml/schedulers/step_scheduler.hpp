// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "scheduler_base.hpp"

namespace ttml::schedulers {

class StepScheduler : public ClosedFormLRScheduler {
public:
    StepScheduler(optimizers::OptimizerBase *optimizer, size_t step_size, float gamma = 0.1f);

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

protected:
    [[nodiscard]] float compute_lr(float base_lr) const override;

    void restore_own_state(const serialization::StateDict &dict) override;

private:
    size_t m_step_size = 0;
    float m_gamma = 0;
};

}  // namespace ttml::schedulers
