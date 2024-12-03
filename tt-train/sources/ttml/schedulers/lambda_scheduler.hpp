// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <utility>

#include "core/not_null.hpp"
#include "scheduler_base.hpp"

// Assuming the necessary includes for core::not_null and optimizers::OptimizerBase
namespace ttml::schedulers {
class LambdaScheduler : public LRSchedulerBase {
public:
    explicit LambdaScheduler(
        core::not_null<optimizers::OptimizerBase *> optimizer, std::function<float(int)> lr_lambda);

    void step() override;

    [[nodiscard]] float get_last_lr() const override;

    [[nodiscard]] float get_current_lr() const override;

private:
    std::function<float(int)> m_lr_lambda;
    int m_last_epoch = -1;
    float m_base_lr = 0.0F;
    float m_last_lr = 0.0F;
};
}  // namespace ttml::schedulers
