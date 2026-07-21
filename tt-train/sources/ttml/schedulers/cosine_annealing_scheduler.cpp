// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cosine_annealing_scheduler.hpp"

#include <tt_stl/assert.hpp>

#include "optimizers/optimizer_base.hpp"

namespace ttml::schedulers {

CosineAnnealingScheduler::CosineAnnealingScheduler(optimizers::OptimizerBase* optimizer, size_t T_max, float eta_min) :
    LRSchedulerBase(optimizer),
    m_T_max(T_max),
    m_eta_min(eta_min),
    m_base_lr(optimizer->get_lr()),
    m_last_step(0),
    m_last_lr(optimizer->get_lr()) {
    TT_FATAL(T_max > 0, "T_max = {} must be greater than zero.", T_max);
}

void CosineAnnealingScheduler::step() {
    m_last_step += 1;

    // Mirror PyTorch CosineAnnealingLR: cos naturally cycles every 2*T_max steps,
    // so no explicit modulo is needed.  Using % T_max would wrongly restart at
    // step T_max (where the LR should be eta_min, not base_lr).
    float cos_inner = static_cast<float>(M_PI) * static_cast<float>(m_last_step) / static_cast<float>(m_T_max);
    float new_lr = m_eta_min + 0.5F * (m_base_lr - m_eta_min) * (1.F + std::cos(cos_inner));

    get_optimizer()->set_lr(new_lr);
    m_last_lr = new_lr;
}

float CosineAnnealingScheduler::get_last_lr() const {
    return m_last_lr;
}

float CosineAnnealingScheduler::get_current_lr() const {
    return get_optimizer()->get_lr();
}

serialization::StateDict CosineAnnealingScheduler::get_state_dict() const {
    serialization::StateDict res;
    res["m_last_step"] = m_last_step;
    res["m_last_lr"] = m_last_lr;
    res["m_base_lr"] = m_base_lr;
    res["m_T_max"] = m_T_max;
    res["m_eta_min"] = m_eta_min;
    return res;
}

void CosineAnnealingScheduler::set_state_dict(const serialization::StateDict& dict) {
    m_last_step = serialization::get_value_type<size_t>(dict, "m_last_step");
    m_last_lr = serialization::get_value_type<float>(dict, "m_last_lr");
    m_base_lr = serialization::get_value_type<float>(dict, "m_base_lr");
    m_T_max = serialization::get_value_type<size_t>(dict, "m_T_max");
    m_eta_min = serialization::get_value_type<float>(dict, "m_eta_min");
}

}  // namespace ttml::schedulers
