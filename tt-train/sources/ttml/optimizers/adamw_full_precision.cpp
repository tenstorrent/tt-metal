// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_full_precision.hpp"

#include <fmt/format.h>

#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

std::string AdamWFullPrecision::get_name() const {
    return "AdamWFullPrecision";
}

AdamWFullPrecision::AdamWFullPrecision(
    ttml::serialization::NamedParameters parameters, const AdamWFullPrecisionConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            // Create fp32 master weights from the initial bf16 weights
            auto bf16_weights = tensor_ptr->get_value(autograd::PreferredPrecision::FULL);
            auto fp32_master = ttnn::typecast(bf16_weights, tt::tt_metal::DataType::FLOAT32);
            m_master_weights.emplace(name, autograd::create_tensor(fp32_master, /* requires_grad */ false));

            // Create fp32 momentum buffers
            m_exp_avg.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(fp32_master),
                    /* requires_grad */ false));
            m_exp_avg_sq.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(fp32_master),
                    /* requires_grad */ false));
        }
    }
    if (m_config.amsgrad) {
        init_max_exp_avg_sq();
    }
}

void AdamWFullPrecision::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void AdamWFullPrecision::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;
    // Update beta powers: multiply by beta values each step instead of computing pow(beta, step)
    m_beta1_pow *= m_config.beta1;
    m_beta2_pow *= m_config.beta2;

    for (const auto& [name, theta_ptr] : m_parameters) {
        if (!theta_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = theta_ptr->get_grad();

        const auto& master_weights = m_master_weights.at(name)->get_value(autograd::PreferredPrecision::FULL);
        const auto& exp_avg = m_exp_avg.at(name)->get_value(autograd::PreferredPrecision::FULL);
        const auto& exp_avg_sq = m_exp_avg_sq.at(name)->get_value(autograd::PreferredPrecision::FULL);

        std::optional<ttnn::Tensor> max_exp_avg_sq;
        if (m_config.amsgrad) {
            max_exp_avg_sq = m_max_exp_avg_sq.at(name)->get_value(autograd::PreferredPrecision::FULL);
        }

        // Call the metal kernel - updates master_weights, exp_avg, exp_avg_sq in place
        ttml::metal::adamw(
            master_weights,
            gradients,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            m_config.lr,
            m_config.beta1,
            m_config.beta2,
            m_beta1_pow,
            m_beta2_pow,
            m_config.epsilon,
            m_config.weight_decay);

        // Convert updated fp32 master weights back to bf16 visible weights
        auto updated_bf16_weights = ttnn::typecast(master_weights, tt::tt_metal::DataType::BFLOAT16);
        theta_ptr->set_value(updated_bf16_weights);
    }
}

serialization::StateDict AdamWFullPrecision::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    dict["master_weights"] = m_master_weights;
    dict["exp_avg"] = m_exp_avg;
    dict["exp_avg_sq"] = m_exp_avg_sq;
    dict["amsgrad"] = m_config.amsgrad;
    if (m_config.amsgrad) {
        dict["max_exp_avg_sq"] = m_max_exp_avg_sq;
    }
    return dict;
}

void AdamWFullPrecision::set_state_dict(const serialization::StateDict& dict) {
    set_steps(serialization::get_value_type<size_t>(dict, "steps"));
    m_master_weights = std::get<serialization::NamedParameters>(dict.at("master_weights"));
    m_exp_avg = std::get<serialization::NamedParameters>(dict.at("exp_avg"));
    m_exp_avg_sq = std::get<serialization::NamedParameters>(dict.at("exp_avg_sq"));

    const bool amsgrad =
        dict.contains("amsgrad") ? serialization::get_value_type<bool>(dict, "amsgrad") : m_config.amsgrad;
    if (amsgrad && dict.contains("max_exp_avg_sq")) {
        m_config.amsgrad = true;
        m_max_exp_avg_sq = std::get<serialization::NamedParameters>(dict.at("max_exp_avg_sq"));
    } else {
        set_amsgrad(amsgrad);
    }
}

size_t AdamWFullPrecision::get_steps() const {
    return m_steps;
}

void AdamWFullPrecision::set_steps(size_t steps) {
    this->m_beta1_pow = std::pow(m_config.beta1, steps);
    this->m_beta2_pow = std::pow(m_config.beta2, steps);
    this->m_steps = steps;
}

float AdamWFullPrecision::get_lr() const {
    return m_config.lr;
}

void AdamWFullPrecision::set_lr(float lr) {
    m_config.lr = lr;
}

float AdamWFullPrecision::get_beta1() const {
    return m_config.beta1;
}

void AdamWFullPrecision::set_beta1(float beta1) {
    m_config.beta1 = beta1;
}

float AdamWFullPrecision::get_beta2() const {
    return m_config.beta2;
}

void AdamWFullPrecision::set_beta2(float beta2) {
    m_config.beta2 = beta2;
}

float AdamWFullPrecision::get_epsilon() const {
    return m_config.epsilon;
}

void AdamWFullPrecision::set_epsilon(float epsilon) {
    m_config.epsilon = epsilon;
}

float AdamWFullPrecision::get_weight_decay() const {
    return m_config.weight_decay;
}

void AdamWFullPrecision::set_weight_decay(float weight_decay) {
    m_config.weight_decay = weight_decay;
}

bool AdamWFullPrecision::get_amsgrad() const {
    return m_config.amsgrad;
}

void AdamWFullPrecision::set_amsgrad(bool amsgrad) {
    if (m_config.amsgrad == amsgrad) {
        return;
    }
    m_config.amsgrad = amsgrad;
    amsgrad ? init_max_exp_avg_sq() : m_max_exp_avg_sq.clear();
}

const ttml::serialization::NamedParameters& AdamWFullPrecision::get_master_weights() const {
    return m_master_weights;
}

void AdamWFullPrecision::init_max_exp_avg_sq() {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            // Use fp32 for max_exp_avg_sq
            auto fp32_master = m_master_weights.at(name)->get_value(autograd::PreferredPrecision::FULL);
            m_max_exp_avg_sq.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(fp32_master),
                    /* requires_grad */ false));
        }
    }
}
}  // namespace ttml::optimizers
