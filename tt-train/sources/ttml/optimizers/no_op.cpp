// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "no_op.hpp"

#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

NoOp::NoOp(ttml::serialization::NamedParameters parameters) : OptimizerBase(std::move(parameters)) {
}

void NoOp::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void NoOp::step() {
    m_steps++;
}

serialization::StateDict NoOp::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    return dict;
}

void NoOp::set_state_dict(const serialization::StateDict& dict) {
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t NoOp::get_steps() const {
    return m_steps;
}

void NoOp::set_steps(size_t steps) {
    this->m_steps = steps;
}

float NoOp::get_lr() const {
    return 0.0f;
}

void NoOp::set_lr(float lr) {
}

}  // namespace ttml::optimizers
