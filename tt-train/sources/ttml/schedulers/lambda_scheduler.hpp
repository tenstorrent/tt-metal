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

protected:
    [[nodiscard]] float compute_lr(float base_lr) const override;

private:
    std::function<float(int)> m_lr_lambda;
};
}  // namespace ttml::schedulers
