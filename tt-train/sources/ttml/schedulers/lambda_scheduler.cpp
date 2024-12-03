#include "lambda_scheduler.hpp"

namespace ttml::schedulers {

LambdaScheduler::LambdaScheduler(
    core::not_null<optimizers::OptimizerBase *> optimizer, std::function<float(int)> lr_lambda) :
    LRSchedulerBase(optimizer),
    m_lr_lambda(std::move(lr_lambda)),
    m_last_epoch(-1),
    m_base_lr(optimizer->get_learning_rate()),
    m_last_lr(optimizer->get_learning_rate()) {
}
void LambdaScheduler::step() {
    m_last_epoch += 1;
    float lr_factor = m_lr_lambda(m_last_epoch);
    float new_lr = m_base_lr * lr_factor;
    get_optimizer()->set_learning_rate(new_lr);
    m_last_lr = new_lr;
}
float LambdaScheduler::get_last_lr() const {
    return m_last_lr;
}
float LambdaScheduler::get_current_lr() const {
    return get_optimizer()->get_learning_rate();
}
}  // namespace ttml::schedulers
