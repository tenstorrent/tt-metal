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
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

const std::string kFirstMoment = "first_moment/";
const std::string kSecondMoment = "second_moment/";

}  // namespace

namespace ttml::optimizers {

MorehAdamW::MorehAdamW(autograd::NamedParameters parameters, const AdamWConfig& config) :
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

        const auto& gradients = tensor_ptr->get_grad();
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

[[nodiscard]] autograd::NamedParameters MorehAdamW::get_state_dict() const {
    autograd::NamedParameters state_dict;
    for (const auto& [key, first_moment] : m_first_moment) {
        state_dict.emplace(kFirstMoment + key, first_moment);
    }

    for (const auto& [key, second_moment] : m_second_moment) {
        state_dict.emplace(kSecondMoment + key, second_moment);
    }

    return state_dict;
}

void MorehAdamW::set_state_dict(const autograd::NamedParameters& dict) {
    for (const auto& [key, tensor] : dict) {
        if (key.starts_with(kFirstMoment)) {
            m_first_moment[key.substr(kFirstMoment.size())] = tensor;
        } else if (key.starts_with(kSecondMoment)) {
            m_second_moment[key.substr(kSecondMoment.size())] = tensor;
        } else {
            throw std::runtime_error(fmt::format("AdamW: Invalid key in state dict. Key = {}", key));
        }
    }
}

[[nodiscard]] size_t MorehAdamW::get_steps() const {
    return m_steps;
}

void MorehAdamW::set_steps(size_t steps) {
    m_steps = steps;
}

AdamW::AdamW(autograd::NamedParameters parameters, const AdamWConfig& config) :
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

        const auto& gradients = tensor_ptr->get_grad();
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
        tensor_ptr->set_value(ttnn::subtract(
            tensor_ptr->get_value(autograd::PreferredPrecision::FULL),
            ttnn_fixed::divide(
                ttnn::multiply(first_moment_hat, m_config.lr),
                ttnn::add(ttnn::sqrt(second_moment_hat), m_config.epsilon))));
    }
}

[[nodiscard]] autograd::NamedParameters AdamW::get_state_dict() const {
    autograd::NamedParameters state_dict;
    for (const auto& [key, first_moment] : m_first_moment) {
        state_dict.emplace(kFirstMoment + key, first_moment);
    }

    for (const auto& [key, second_moment] : m_second_moment) {
        state_dict.emplace(kSecondMoment + key, second_moment);
    }

    return state_dict;
}

void AdamW::set_state_dict(const autograd::NamedParameters& dict) {
    for (const auto& [key, tensor] : dict) {
        if (key.starts_with(kFirstMoment)) {
            m_first_moment[key.substr(kFirstMoment.size())] = tensor;
        } else if (key.starts_with(kSecondMoment)) {
            m_second_moment[key.substr(kSecondMoment.size())] = tensor;
        } else {
            throw std::runtime_error(fmt::format("AdamW: Invalid key in state dict. Key = {}", key));
        }
    }
}

[[nodiscard]] size_t AdamW::get_steps() const {
    return m_steps;
}

void AdamW::set_steps(size_t steps) {
    m_steps = steps;
}

}  // namespace ttml::optimizers
