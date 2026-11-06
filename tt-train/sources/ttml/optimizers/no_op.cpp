// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "no_op.hpp"

#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

NoOp::NoOp(ttml::serialization::NamedParameters parameters) : OptimizerBase(std::move(parameters)) {
    init_param_groups({{"lr", 0.0F}}, {});
}

void NoOp::zero_grad() {
    install_zero_gradients(m_parameters);
}

void NoOp::step() {
    m_steps++;
}

serialization::StateDict NoOp::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    save_param_groups(dict);
    return dict;
}

void NoOp::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t NoOp::get_steps() const {
    return m_steps;
}

void NoOp::set_steps(size_t steps) {
    this->m_steps = steps;
}
}  // namespace ttml::optimizers
