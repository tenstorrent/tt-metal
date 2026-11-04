// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cosine_annealing_scheduler.hpp"

#include <tt_stl/assert.hpp>

namespace ttml::schedulers {

CosineAnnealingScheduler::CosineAnnealingScheduler(optimizers::OptimizerBase* optimizer, size_t T_max, float eta_min) :
    ClosedFormLRScheduler(optimizer), m_T_max(T_max), m_eta_min(eta_min) {
    TT_FATAL(T_max > 0, "T_max = {} must be greater than zero.", T_max);

    // At step 0 the cosine factor is 1, so every group's LR is (re)set to its base LR.
    apply_lrs();
}

float CosineAnnealingScheduler::compute_lr(float base_lr) const {
    // Mirror PyTorch CosineAnnealingLR: cos naturally cycles every 2*T_max steps,
    // so no explicit modulo is needed.  Using % T_max would wrongly restart at
    // step T_max (where the LR should be eta_min, not base_lr).
    float cos_inner = static_cast<float>(M_PI) * static_cast<float>(m_last_step) / static_cast<float>(m_T_max);
    return m_eta_min + 0.5F * (base_lr - m_eta_min) * (1.F + std::cos(cos_inner));
}

serialization::StateDict CosineAnnealingScheduler::get_state_dict() const {
    auto res = ClosedFormLRScheduler::get_state_dict();
    res["m_T_max"] = m_T_max;
    res["m_eta_min"] = m_eta_min;
    return res;
}

void CosineAnnealingScheduler::restore_own_state(const serialization::StateDict& dict) {
    const auto t_max = serialization::get_value_type<size_t>(dict, "m_T_max");
    const auto eta_min = serialization::get_value_type<float>(dict, "m_eta_min");
    m_T_max = t_max;
    m_eta_min = eta_min;
}

}  // namespace ttml::schedulers
