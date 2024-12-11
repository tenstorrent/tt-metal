// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scheduler_base.hpp"

namespace ttml::schedulers {

class LinearScheduler : public LRSchedulerBase {
public:
    LinearScheduler(optimizers::OptimizerBase *optimizer, float start_factor, float end_factor, size_t total_steps);

    void step() override;

    [[nodiscard]] float get_last_lr() const override;

    [[nodiscard]] float get_current_lr() const override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict &dict) override;

private:
    float m_base_lr = 0.F;
    float m_start_factor = 0.F;
    float m_end_factor = 0.F;
    int m_total_steps = 0;
    size_t m_last_step = 0;
    float m_last_lr = 0.F;
};
}  // namespace ttml::schedulers
