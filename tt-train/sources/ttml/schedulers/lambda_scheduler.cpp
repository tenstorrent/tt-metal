// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lambda_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"
namespace ttml::schedulers {

LambdaScheduler::LambdaScheduler(optimizers::OptimizerBase *optimizer, std::function<float(int)> lr_lambda) :
    LRSchedulerBase(optimizer), m_lr_lambda(std::move(lr_lambda)), m_base_lr(optimizer->get_initial_lr()) {
    // Mirror PyTorch's LambdaLR, which applies lr_lambda(0) at construction.
    update_lr(m_base_lr * m_lr_lambda(0));
}
void LambdaScheduler::step() {
    m_last_step += 1;
    float lr_factor = m_lr_lambda(m_last_step);
    update_lr(m_base_lr * lr_factor);
}
// NOTE: ``m_lr_lambda`` is a type-erased ``std::function`` and cannot be
// introspected (no equivalent of Python's callable ``__dict__``), so it is not
// part of the saved state. Callers must reconstruct the scheduler with the
// same ``lr_lambda`` before restoring state.
void LambdaScheduler::set_state_dict(const serialization::StateDict &dict) {
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_base_lr = serialization::get_value_type<float>(dict, "m_base_lr");
    // Restore the live LR (see LRSchedulerBase::update_lr for rationale).
    update_lr(serialization::get_value_type<float>(dict, "m_last_lr"));
}
serialization::StateDict LambdaScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    res["m_base_lr"] = m_base_lr;
    return res;
};
}  // namespace ttml::schedulers
