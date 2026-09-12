// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizer_base.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

OptimizerBase::OptimizerBase(serialization::NamedParameters&& parameters) : m_parameters(std::move(parameters)) {
}

float OptimizerBase::get_initial_lr() {
    if (!m_initial_lr.has_value()) {
        m_initial_lr = get_lr();
    }
    return *m_initial_lr;
}

void OptimizerBase::save_initial_lr(serialization::StateDict& dict) const {
    // If no scheduler has recorded a base LR yet, the current LR is the base.
    dict["initial_lr"] = m_initial_lr.value_or(get_lr());
}

void OptimizerBase::restore_initial_lr(const serialization::StateDict& dict) {
    m_initial_lr = serialization::get_value_type<float>(dict, "initial_lr");
}

void OptimizerBase::print_stats() const {
    fmt::print("\n\nOptimization parameters values and gradients:\n");
    for (const auto& [name, tensor] : m_parameters) {
        core::print_tensor_stats(tensor->get_value(), fmt::format("{}/value", name));
        if (tensor->is_grad_initialized()) {
            core::print_tensor_stats(tensor->get_grad(), fmt::format("{}/gradient", name));
        }
    }
    fmt::print("=================================================\n");
}

}  // namespace ttml::optimizers
