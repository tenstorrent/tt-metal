// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scheduler_base.hpp"

namespace ttml::schedulers {

class LinearScheduler : public ClosedFormLRScheduler {
public:
    LinearScheduler(optimizers::OptimizerBase *optimizer, float start_factor, float end_factor, size_t total_steps);

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

protected:
    [[nodiscard]] float compute_lr(float base_lr) const override;

    void restore_own_state(const serialization::StateDict &dict) override;

private:
    float m_start_factor = 0.F;
    float m_end_factor = 0.F;
    int m_total_steps = 0;
};
}  // namespace ttml::schedulers
