// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "lambda_scheduler.hpp"

#include "optimizers/optimizer_base.hpp"
namespace ttml::schedulers {

LambdaScheduler::LambdaScheduler(optimizers::OptimizerBase *optimizer, std::function<float(int)> lr_lambda) :
    LRSchedulerBase(optimizer),
    m_lr_lambda(std::move(lr_lambda)),
    m_last_step(0),
    m_base_lr(optimizer->get_lr()),
    m_last_lr(optimizer->get_lr()) {
}
void LambdaScheduler::step() {
    m_last_step += 1;
    float lr_factor = m_lr_lambda(m_last_step);
    float new_lr = m_base_lr * lr_factor;
    get_optimizer()->set_lr(new_lr);
    m_last_lr = new_lr;
}
float LambdaScheduler::get_last_lr() const {
    return m_last_lr;
}
float LambdaScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}
void LambdaScheduler::set_state_dict(const serialization::StateDict &dict) {
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_last_lr = serialization::get_value_type<float>(dict, "m_last_lr");
}
serialization::StateDict LambdaScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    return res;
};
}  // namespace ttml::schedulers
