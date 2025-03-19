// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw.hpp"

#include "autograd/autocast_tensor.hpp"
#include "autograd/module_base.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
namespace {

const std::string kFirstMoment = "first_moment";
const std::string kSecondMoment = "second_moment";
const std::string kKahanCompensation = "kahan_compensation";
const std::string kSteps = "steps";
}  // namespace

namespace ttml::optimizers {

MorehAdamW::MorehAdamW(serialization::NamedParameters parameters, const AdamWConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    if (m_config.use_kahan_summation) {
        throw std::runtime_error("MorehAdamW: Kahan summation is not supported. Use default AdamW instead.");
    }

    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
            m_second_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
        }
    }
}

void MorehAdamW::zero_grad() {
    for (auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            // setting gradients to not initialized tensor
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}

void MorehAdamW::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;
    for (auto& [key, first_moment_ptr] : m_first_moment) {
        const auto& tensor_ptr = m_parameters.at(key);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }
        auto& second_moment_ptr = m_second_moment.at(key);
        const auto& first_moment = first_moment_ptr->get_value(autograd::PreferredPrecision::FULL);
        const auto& second_moment = second_moment_ptr->get_value(autograd::PreferredPrecision::FULL);

        auto gradients = tensor_ptr->get_grad();

        auto output_tensor = tensor_ptr->get_value(autograd::PreferredPrecision::FULL);
        ttnn::moreh_adamw(
            tensor_ptr->get_value(autograd::PreferredPrecision::FULL),
            gradients,
            first_moment,
            second_moment,
            m_config.lr,
            m_config.beta1,
            m_config.beta2,
            m_config.epsilon,
            m_config.weight_decay,
            m_steps,
            /* amsgrad */ false,
            /* max_exp_avg_sq_in */ std::nullopt,
            /* param_out */ output_tensor,
            /* exp_avg_out */ first_moment,
            /* exp_avg_sq_out */ second_moment,
            /* max_exp_avg_sq_out */ std::nullopt,
            /* memory_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        tensor_ptr->set_value(output_tensor);
        first_moment_ptr->set_value(first_moment);
        second_moment_ptr->set_value(second_moment);
    }
}

[[nodiscard]] serialization::StateDict MorehAdamW::get_state_dict() const {
    serialization::StateDict state_dict;
    state_dict[kFirstMoment] = m_first_moment;
    state_dict[kSecondMoment] = m_second_moment;
    state_dict[kSteps] = m_steps;

    return state_dict;
}

void MorehAdamW::set_state_dict(const serialization::StateDict& dict) {
    m_first_moment = std::get<serialization::NamedParameters>(dict.at(kFirstMoment));
    m_second_moment = std::get<serialization::NamedParameters>(dict.at(kSecondMoment));
    m_steps = serialization::get_value_type<size_t>(dict, kSteps);
}

[[nodiscard]] size_t MorehAdamW::get_steps() const {
    return m_steps;
}

void MorehAdamW::set_steps(size_t steps) {
    m_steps = steps;
}

float MorehAdamW::get_lr() const {
    return m_config.lr;
}
void MorehAdamW::set_lr(float lr) {
    m_config.lr = lr;
}

AdamW::AdamW(serialization::NamedParameters parameters, const AdamWConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
            m_second_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                    /* requires_grad */ false));
            if (m_config.use_kahan_summation) {
                m_kahan_compensation.emplace(
                    key,
                    autograd::create_tensor(
                        core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::FULL)),
                        /* requires_grad */ false));
            }
        }
    }
}

void AdamW::zero_grad() {
    for (auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            // setting gradients to not initialized tensor
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}

void AdamW::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;
    for (auto& [key, first_moment_ptr] : m_first_moment) {
        const auto& tensor_ptr = m_parameters.at(key);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }
        auto& second_moment_ptr = m_second_moment.at(key);
        auto first_moment = first_moment_ptr->get_value(autograd::PreferredPrecision::FULL);
        auto second_moment = second_moment_ptr->get_value(autograd::PreferredPrecision::FULL);

        auto gradients = tensor_ptr->get_grad();

        if (m_config.weight_decay != 0.0F) {
            auto weight_decay_update = ttnn::multiply(
                tensor_ptr->get_value(autograd::PreferredPrecision::FULL), m_config.weight_decay * m_config.lr);
            // weights -= weight_decay * lr * weights
            tensor_ptr->set_value(
                ttnn::subtract(tensor_ptr->get_value(autograd::PreferredPrecision::FULL), weight_decay_update));
        }

        // first moment = beta1 * first moment + (1 - beta1) * gradients
        first_moment =
            ttnn::add(ttnn::multiply(first_moment, m_config.beta1), ttnn::multiply(gradients, 1.F - m_config.beta1));
        // second moment = beta2 * second moment + (1 - beta2) * gradients^2
        second_moment = ttnn::add(
            ttnn::multiply(second_moment, m_config.beta2),
            ttnn::multiply(ttnn::square(gradients), 1.F - m_config.beta2));
        // first_moment_hat = first_moment / (1 - beta1^steps)
        auto first_moment_hat = ttnn::multiply(first_moment, 1.F / (1.F - std::pow(m_config.beta1, m_steps)));
        // second_moment_hat = second_moment / (1 - beta2^steps)
        auto second_moment_hat = ttnn::multiply(second_moment, 1.F / (1.F - std::pow(m_config.beta2, m_steps)));
        // weights -= lr * first_moment_hat / (sqrt(second_moment_hat) + epsilon)
        first_moment_ptr->set_value(first_moment);
        second_moment_ptr->set_value(second_moment);

        auto update_tensor = ttnn_fixed::divide(
            ttnn::multiply(first_moment_hat, -m_config.lr), ttnn::add(ttnn::sqrt(second_moment_hat), m_config.epsilon));

        if (!m_config.use_kahan_summation) {
            tensor_ptr->set_value(ttnn::add(tensor_ptr->get_value(autograd::PreferredPrecision::FULL), update_tensor));
        } else {
            auto value_tensor = tensor_ptr->get_value(autograd::PreferredPrecision::FULL);

            const auto& kahan_compensation_ptr = m_kahan_compensation.at(key);
            // A running compensation for lost low-order bits
            auto compensation_tensor = kahan_compensation_ptr->get_value(autograd::PreferredPrecision::FULL);
            // Adjust the update with the compensation
            auto adjusted_update = ttnn::subtract(update_tensor, compensation_tensor);
            // Update the value with the adjusted update
            auto result = ttnn::add(value_tensor, adjusted_update);
            // (result - value_tensor) cancels the high-order part of adjusted_update;
            // subtracting adjusted_update recovers negative (low part of adjusted_update)
            compensation_tensor = ttnn::subtract(ttnn::subtract(result, value_tensor), adjusted_update);

            tensor_ptr->set_value(result);
            kahan_compensation_ptr->set_value(compensation_tensor);
        }
    }
}

[[nodiscard]] serialization::StateDict AdamW::get_state_dict() const {
    serialization::StateDict state_dict;
    state_dict[kFirstMoment] = m_first_moment;
    state_dict[kSecondMoment] = m_second_moment;
    state_dict[kKahanCompensation] = m_kahan_compensation;
    state_dict[kSteps] = m_steps;

    return state_dict;
}

void AdamW::set_state_dict(const serialization::StateDict& dict) {
    m_first_moment = std::get<serialization::NamedParameters>(dict.at(kFirstMoment));
    m_second_moment = std::get<serialization::NamedParameters>(dict.at(kSecondMoment));
    m_kahan_compensation = std::get<serialization::NamedParameters>(dict.at(kKahanCompensation));
    m_steps = serialization::get_value_type<size_t>(dict, kSteps);
}

[[nodiscard]] size_t AdamW::get_steps() const {
    return m_steps;
}

void AdamW::set_steps(size_t steps) {
    m_steps = steps;
}

float AdamW::get_lr() const {
    return m_config.lr;
}
void AdamW::set_lr(float lr) {
    m_config.lr = lr;
}
}  // namespace ttml::optimizers
