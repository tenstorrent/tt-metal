// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "scheduler_base.hpp"

namespace ttml::schedulers {
class LambdaScheduler : public LRSchedulerBase {
public:
    explicit LambdaScheduler(optimizers::OptimizerBase *optimizer, std::function<float(int)> lr_lambda);

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;

    void set_state_dict(const serialization::StateDict &dict) override;

private:
    std::function<float(int)> m_lr_lambda;
    size_t m_last_step = 0;
    float m_base_lr = 0.0F;
};
}  // namespace ttml::schedulers
