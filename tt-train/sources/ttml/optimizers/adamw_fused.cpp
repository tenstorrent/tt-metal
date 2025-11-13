// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_fused.hpp"

#include <fmt/format.h>

#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

AdamWFused::AdamWFused(ttml::serialization::NamedParameters parameters, const AdamWFusedConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    if (m_config.weight_decay != 0.0F) {
        throw std::runtime_error("AdamWFused: weight_decay is not supported. Use default AdamW instead.");
    }
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
            m_second_moment.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
        }
    }
}

void AdamWFused::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void AdamWFused::step() {
    if (m_config.weight_decay != 0.0F) {
        throw std::runtime_error("AdamWFused: weight_decay is not supported. Use default AdamW instead.");
    }
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;

    for (const auto& [name, theta_ptr] : m_parameters) {
        if (!theta_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = theta_ptr->get_grad();
        auto param = theta_ptr->get_value(autograd::PreferredPrecision::FULL);

        auto it_first = m_first_moment.find(name);
        auto it_second = m_second_moment.find(name);

        if (it_first == m_first_moment.end()) {
            auto buf = autograd::create_tensor(
                core::zeros_like(param),
                /* requires_grad */ false);
            it_first = m_first_moment.emplace(name, std::move(buf)).first;
        }

        if (it_second == m_second_moment.end()) {
            auto buf = autograd::create_tensor(
                core::zeros_like(param),
                /* requires_grad */ false);
            it_second = m_second_moment.emplace(name, std::move(buf)).first;
        }

        auto first_moment = it_first->second->get_value(autograd::PreferredPrecision::FULL);
        auto second_moment = it_second->second->get_value(autograd::PreferredPrecision::FULL);

        ttml::metal::adamw_fused(
            param,
            gradients,
            first_moment,
            second_moment,
            m_config.lr,
            m_config.beta1,
            m_config.beta2,
            m_config.epsilon,
            m_config.weight_decay,
            m_steps);
    }
}

serialization::StateDict AdamWFused::get_state_dict() const {
    serialization::StateDict dict;
    dict["first_moment"] = m_first_moment;
    dict["second_moment"] = m_second_moment;
    dict["steps"] = m_steps;
    return dict;
}

void AdamWFused::set_state_dict(const serialization::StateDict& dict) {
    m_first_moment = std::get<serialization::NamedParameters>(dict.at("first_moment"));
    m_second_moment = std::get<serialization::NamedParameters>(dict.at("second_moment"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
}

size_t AdamWFused::get_steps() const {
    return m_steps;
}

void AdamWFused::set_steps(size_t steps) {
    this->m_steps = steps;
}

float AdamWFused::get_lr() const {
    return m_config.lr;
}

void AdamWFused::set_lr(float lr) {
    m_config.lr = lr;
}

float AdamWFused::get_beta1() const {
    return m_config.beta1;
}

void AdamWFused::set_beta1(float beta1) {
    m_config.beta1 = beta1;
}

float AdamWFused::get_beta2() const {
    return m_config.beta2;
}

void AdamWFused::set_beta2(float beta2) {
    m_config.beta2 = beta2;
}

float AdamWFused::get_epsilon() const {
    return m_config.epsilon;
}

void AdamWFused::set_epsilon(float epsilon) {
    m_config.epsilon = epsilon;
}

float AdamWFused::get_weight_decay() const {
    return m_config.weight_decay;
}

void AdamWFused::set_weight_decay(float weight_decay) {
    m_config.weight_decay = weight_decay;
}

}  // namespace ttml::optimizers
