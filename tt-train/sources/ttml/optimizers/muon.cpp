// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "muon.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/newton_schulz_op.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

Muon::Muon(ttml::serialization::NamedParameters parameters, const MuonConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_momentum_buffer.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
        }
    }
}

void Muon::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void Muon::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (auto& [name, buffer_ptr] : m_momentum_buffer) {
        auto buffer = buffer_ptr->get_value(autograd::PreferredPrecision::FULL);
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = tensor_ptr->get_grad();

        if (m_steps > 0 && m_config.momentum != 0.0F) {
            buffer = ttnn::multiply(buffer, m_config.momentum);
            buffer = ttnn::add(buffer, gradients);
        } else {
            buffer = gradients;
        }

        buffer_ptr->set_value(buffer);

        auto update_direction = ops::newtonschulz5(buffer, m_config.ns_steps, 1e-7f);

        tensor_ptr->set_value(ttnn::subtract(
            tensor_ptr->get_value(autograd::PreferredPrecision::FULL), ttnn::multiply(update_direction, m_config.lr)));
    }
    m_steps++;
}

serialization::StateDict Muon::get_state_dict() const {
    serialization::StateDict dict;
    dict["momentum_buffer"] = m_momentum_buffer;
    dict["steps"] = m_steps;
    return dict;
}

void Muon::set_state_dict(const serialization::StateDict& dict) {
    m_momentum_buffer = std::get<serialization::NamedParameters>(dict.at("momentum_buffer"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t Muon::get_steps() const {
    return m_steps;
}

void Muon::set_steps(size_t steps) {
    this->m_steps = steps;
}

}  // namespace ttml::optimizers
