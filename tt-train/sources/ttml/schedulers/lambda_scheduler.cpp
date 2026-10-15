// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lambda_scheduler.hpp"

namespace ttml::schedulers {

LambdaScheduler::LambdaScheduler(optimizers::OptimizerBase *optimizer, std::function<float(int)> lr_lambda) :
    ClosedFormLRScheduler(optimizer), m_lr_lambda(std::move(lr_lambda)) {
    // Mirror PyTorch's LambdaLR, which applies lr_lambda(0) at construction.
    apply_lrs();
}

float LambdaScheduler::compute_lr(float base_lr) const {
    return base_lr * m_lr_lambda(static_cast<int>(m_last_step));
}

}  // namespace ttml::schedulers
