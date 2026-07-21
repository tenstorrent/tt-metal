// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "scheduler_base.hpp"

namespace ttml::schedulers {

// Cosine annealing: decays LR from base_lr to eta_min following a cosine curve
// over T_max steps, then optionally restarts (SGDR-style) if step() is called
// beyond T_max.  eta_min defaults to 0.
class CosineAnnealingScheduler : public LRSchedulerBase {
public:
    CosineAnnealingScheduler(optimizers::OptimizerBase* optimizer, size_t T_max, float eta_min = 0.F);

    void step() override;

    [[nodiscard]] float get_last_lr() const override;

    [[nodiscard]] float get_current_lr() const override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

private:
    size_t m_T_max = 0;
    float m_eta_min = 0.F;
    float m_base_lr = 0.F;
    size_t m_last_step = 0;
    float m_last_lr = 0.F;
};

}  // namespace ttml::schedulers
