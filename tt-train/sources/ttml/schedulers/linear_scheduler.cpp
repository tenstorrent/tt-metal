// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_scheduler.hpp"

#include <algorithm>
#include <tt_stl/assert.hpp>

namespace ttml::schedulers {

LinearScheduler::LinearScheduler(
    optimizers::OptimizerBase* optimizer, float start_factor, float end_factor, size_t total_steps) :
    ClosedFormLRScheduler(optimizer),
    m_start_factor(start_factor),
    m_end_factor(end_factor),
    m_total_steps(static_cast<int>(total_steps)) {
    TT_FATAL(total_steps > 0, "total_steps = {} must be greater than zero.", total_steps);
    // Bounds match PyTorch's LinearLR. start_factor == 0 is rejected because
    // the factor is applied at construction, so it would make the first
    // optimizer step a no-op (LR = 0).
    TT_FATAL(
        start_factor > 0.0F && start_factor <= 1.0F,
        "start_factor = {} expected to be greater than 0 and less or equal to 1.",
        start_factor);
    TT_FATAL(end_factor >= 0.0F && end_factor <= 1.0F, "end_factor = {} expected to be between 0 and 1.", end_factor);

    // Mirror PyTorch, which applies start_factor at construction: the LR used
    // before the first step() is already base_lr * start_factor.
    apply_lrs();
}

float LinearScheduler::compute_lr(float base_lr) const {
    float progress = static_cast<float>(m_last_step) / m_total_steps;
    progress = std::min(progress, 1.0f);

    float current_factor = m_start_factor + (m_end_factor - m_start_factor) * progress;
    return base_lr * current_factor;
}

void LinearScheduler::restore_own_state(const serialization::StateDict& dict) {
    const auto start_factor = serialization::get_value_type<float>(dict, "m_start_factor");
    const auto end_factor = serialization::get_value_type<float>(dict, "m_end_factor");
    const auto total_steps = serialization::get_value_type<int>(dict, "m_total_steps");
    m_start_factor = start_factor;
    m_end_factor = end_factor;
    m_total_steps = total_steps;
}

serialization::StateDict LinearScheduler::get_state_dict() const {
    auto res = ClosedFormLRScheduler::get_state_dict();
    res["m_start_factor"] = m_start_factor;
    res["m_end_factor"] = m_end_factor;
    res["m_total_steps"] = m_total_steps;
    return res;
};

}  // namespace ttml::schedulers
