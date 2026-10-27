// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "scheduler_base.hpp"

namespace ttml::schedulers {
class LambdaScheduler : public ClosedFormLRScheduler {
public:
    explicit LambdaScheduler(optimizers::OptimizerBase *optimizer, std::function<float(int)> lr_lambda);

protected:
    [[nodiscard]] float compute_lr(float base_lr) const override;

private:
    // A type-erased std::function cannot be introspected (no equivalent of Python's callable __dict__), so it is not
    // part of the saved state. Callers must reconstruct the scheduler with the same lr_lambda before restoring state.
    std::function<float(int)> m_lr_lambda;
};
}  // namespace ttml::schedulers
