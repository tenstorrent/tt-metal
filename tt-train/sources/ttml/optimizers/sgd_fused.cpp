// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "fmt/base.h"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

SGDFused::SGDFused(ttml::serialization::NamedParameters parameters, const SGDFusedConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    assert(!(m_config.nesterov && m_config.dampening != 0.0) && "Nesterov momentum requires zero dampening");
    assert(!(m_config.nesterov && m_config.momentum <= 0.0) && "Nesterov momentum requires a positive momentum");

    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_theta.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
        }
    }
}

void SGDFused::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGDFused::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (auto& [name, theta_ptr] : m_theta) {
        auto theta = theta_ptr->get_value(autograd::PreferredPrecision::FULL);
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }
        auto gradients = tensor_ptr->get_grad();
        auto output_tensor = tensor_ptr->get_value(autograd::PreferredPrecision::FULL);

#ifdef PRINT_SGD_FUSED_DEBUG_INFO
        fmt::print("{}\n", name);
        tensor_ptr->get_value(autograd::PreferredPrecision::FULL).print();
        fmt::print("momentum before\n");
        theta.print();
#endif

        ttml::metal::sgd_fused(
            tensor_ptr->get_value(autograd::PreferredPrecision::FULL),
            gradients,
            m_config.lr,
            m_config.momentum,
            m_config.dampening,
            m_config.weight_decay,
            m_config.nesterov,
            output_tensor,
            theta,
            theta);
#ifdef PRINT_SGD_FUSED_DEBUG_INFO
        fmt::print("gradient\n");
        gradients.print();
        fmt::print("output parameters\n");
        output_tensor.print();
        fmt::print("momentum after\n");
        theta.print();
        fmt::print("learning rate: {}\n", m_config.lr);
        fmt::print("===================================\n");
#endif
    }
    m_steps++;
}

serialization::StateDict SGDFused::get_state_dict() const {
    serialization::StateDict dict;
    dict["theta"] = m_theta;
    dict["steps"] = m_steps;
    return dict;
}

void SGDFused::set_state_dict(const serialization::StateDict& dict) {
    m_theta = std::get<serialization::NamedParameters>(dict.at("theta"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t SGDFused::get_steps() const {
    return m_steps;
}

void SGDFused::set_steps(size_t steps) {
    this->m_steps = steps;
}

float SGDFused::get_lr() const {
    return m_config.lr;
}

void SGDFused::set_lr(float lr) {
    m_config.lr = lr;
}

}  // namespace ttml::optimizers
