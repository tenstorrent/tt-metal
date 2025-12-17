// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

SGDFused::SGDFused(ttml::serialization::NamedParameters parameters, const SGDFusedConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    validate_config();
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
    validate_config();

    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (const auto& [name, theta_ptr] : m_parameters) {
        if (!theta_ptr->is_grad_initialized()) {
            continue;
        }
        auto gradients = theta_ptr->get_grad();
        auto param = theta_ptr->get_value(autograd::PreferredPrecision::FULL);

        std::optional<ttnn::Tensor> momentum_buffer;
        if (m_config.momentum > 0.0) {
            auto it = m_momentum.find(name);
            if (it == m_momentum.end()) {
                auto buf = autograd::create_tensor(
                    core::zeros_like(param),
                    /* requires_grad */ false);
                it = m_momentum.emplace(name, std::move(buf)).first;
            }
            momentum_buffer = it->second->get_value(autograd::PreferredPrecision::FULL);
        }

        // The first step should not apply dampening
        float dampening = m_steps == 0 ? 0.0f : m_config.dampening;
        ttml::metal::sgd_fused(
            param,
            gradients,
            m_config.lr,
            m_config.momentum,
            dampening,
            m_config.weight_decay,
            m_config.nesterov,
            momentum_buffer);
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

float SGDFused::get_momentum() const {
    return m_config.momentum;
}

void SGDFused::set_momentum(float momentum) {
    m_config.momentum = momentum;
}

float SGDFused::get_dampening() const {
    return m_config.dampening;
}

void SGDFused::set_dampening(float dampening) {
    m_config.dampening = dampening;
}

float SGDFused::get_weight_decay() const {
    return m_config.weight_decay;
}

void SGDFused::set_weight_decay(float weight_decay) {
    m_config.weight_decay = weight_decay;
}

bool SGDFused::get_nesterov() const {
    return m_config.nesterov;
}

void SGDFused::set_nesterov(bool nesterov) {
    m_config.nesterov = nesterov;
}

void SGDFused::validate_config() const {
    if (m_config.nesterov) {
        if (m_config.dampening != 0.0) {
            throw std::runtime_error(
                fmt::format("Nesterov momentum requires zero dampening! dampening={}", m_config.dampening));
        }
        if (m_config.momentum <= 0.0) {
            throw std::runtime_error(
                fmt::format("Nesterov momentum requires a positive momentum! momentum={}", m_config.momentum));
        }
    }
    if (m_config.dampening != 0.0 && m_config.momentum == 0.0) {
        throw std::runtime_error(fmt::format("Dampening requires a positive momentum! momentum={}", m_config.momentum));
    }
}

}  // namespace ttml::optimizers
