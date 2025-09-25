// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

SGDFused::SGDFused(ttml::serialization::NamedParameters parameters, const SGDFusedConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    TT_FATAL(!(m_config.nesterov && m_config.dampening != 0.0), "Nesterov momentum requires zero dampening");
    TT_FATAL(!(m_config.nesterov && m_config.momentum <= 0.0), "Nesterov momentum requires a positive momentum");

    if (m_config.momentum > 0.0) {
        for (const auto& [name, tensor_ptr] : m_parameters) {
            if (tensor_ptr->get_requires_grad()) {
                m_momentum.emplace(
                    name,
                    autograd::create_tensor(
                        core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                        /* requires_grad */ false));
            }
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

    TT_FATAL(!(m_config.nesterov && m_config.dampening != 0.0), "Nesterov momentum requires zero dampening");
    TT_FATAL(!(m_config.nesterov && m_config.momentum <= 0.0), "Nesterov momentum requires a positive momentum");
    const bool use_momentum = (m_config.momentum > 0.0F);

    for (const auto& [name, theta_ptr] : m_parameters) {
        if (!theta_ptr->is_grad_initialized()) {
            continue;
        }
        auto gradients = theta_ptr->get_grad();
        auto param_in = theta_ptr->get_value(autograd::PreferredPrecision::FULL);
        auto param_out = param_in;

        std::optional<ttnn::Tensor> momentum = std::nullopt;
        // momentum buffers are lazily initialized
        if (use_momentum) {
            auto it = m_momentum.find(name);
            if (it == m_momentum.end()) {
                auto buf = autograd::create_tensor(
                    core::zeros_like(param_in),
                    /* requires_grad */ false);
                it = m_momentum.emplace(name, std::move(buf)).first;
            }
            momentum = it->second->get_value(autograd::PreferredPrecision::FULL);
        }

        ttml::metal::sgd_fused(
            param_in,
            gradients,
            m_config.lr,
            m_config.momentum,
            m_config.dampening,
            m_config.weight_decay,
            m_config.nesterov,
            param_out,
            momentum,
            momentum);
    }
    m_steps++;
}

serialization::StateDict SGDFused::get_state_dict() const {
    serialization::StateDict dict;
    dict["momentum"] = m_momentum;
    dict["steps"] = m_steps;
    return dict;
}

void SGDFused::set_state_dict(const serialization::StateDict& dict) {
    m_momentum = std::get<serialization::NamedParameters>(dict.at("momentum"));
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
