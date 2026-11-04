// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_composite.hpp"

#include "autograd/autocast_tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

const std::string kFirstMoment = "first_moment";
const std::string kSecondMoment = "second_moment";
const std::string kMaxExpAvgSq = "max_exp_avg_sq";
const std::string kKahanCompensation = "kahan_compensation";
const std::string kSteps = "steps";
}  // namespace

namespace ttml::optimizers {

MorehAdamW::MorehAdamW(
    serialization::NamedParameters parameters,
    const AdamWCompositeConfig& config,
    std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)) {
    require_bf16_parameters(m_parameters, "MorehAdamW");
    if (config.kahan_summation) {
        throw std::runtime_error("MorehAdamW: Kahan summation is not supported. Use default AdamWComposite instead.");
    }

    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
            m_second_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
        }
    }
    init_param_groups(param_group_defaults(config), std::move(param_groups));
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
    for (const auto& group : m_param_groups) {
        const float lr = group.options.at("lr");
        const float beta1 = group.options.at("beta1");
        const float beta2 = group.options.at("beta2");
        const float epsilon = group.options.at("epsilon");
        const float weight_decay = group.options.at("weight_decay");

        for (const auto& key : group.params) {
            const auto first_moment_it = m_first_moment.find(key);
            if (first_moment_it == m_first_moment.end()) {
                continue;  // not trainable
            }
            auto& first_moment_ptr = first_moment_it->second;
            const auto& tensor_ptr = m_parameters.at(key);
            if (!tensor_ptr->is_grad_initialized()) {
                continue;
            }
            auto& second_moment_ptr = m_second_moment.at(key);
            // moreh_adamw writes the parameter and the moments in place, through its output tensors.
            auto param = tensor_ptr->get_value_for_update(autograd::PreferredPrecision::HALF);
            auto first_moment = first_moment_ptr->get_value_for_update(autograd::PreferredPrecision::HALF);
            auto second_moment = second_moment_ptr->get_value_for_update(autograd::PreferredPrecision::HALF);

            auto gradients = tensor_ptr->get_grad();

            // By value: the op below relabels its outputs (see optimizers::restore_topology).
            const auto topology = tensor_ptr->get_value(autograd::PreferredPrecision::HALF).tensor_topology();

            ttnn::moreh_adamw(
                param.tensor(),
                gradients,
                first_moment.tensor(),
                second_moment.tensor(),
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                m_steps,
                /* amsgrad */ false,
                /* max_exp_avg_sq_in */ std::nullopt,
                /* param_out */ param.tensor(),
                /* exp_avg_out */ first_moment.tensor(),
                /* exp_avg_sq_out */ second_moment.tensor(),
                /* max_exp_avg_sq_out */ std::nullopt,
                /* memory_config */ std::nullopt,
                /* compute_kernel_config */ core::ComputeKernelConfig::precise());
            restore_topology({tensor_ptr, first_moment_ptr, second_moment_ptr}, topology);
        }
    }
}

[[nodiscard]] serialization::StateDict MorehAdamW::get_state_dict() const {
    serialization::StateDict state_dict;
    state_dict[kFirstMoment] = m_first_moment;
    state_dict[kSecondMoment] = m_second_moment;
    state_dict[kSteps] = m_steps;
    save_param_groups(state_dict);

    return state_dict;
}

void MorehAdamW::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
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

AdamWComposite::AdamWComposite(
    serialization::NamedParameters parameters,
    const AdamWCompositeConfig& config,
    std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)), m_amsgrad(config.amsgrad), m_kahan_summation(config.kahan_summation) {
    require_bf16_parameters(m_parameters, "AdamWComposite");
    for (const auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_first_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
            m_second_moment.emplace(
                key,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
            if (m_amsgrad) {
                m_max_exp_avg_sq.emplace(
                    key,
                    autograd::create_tensor(
                        core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                        /* requires_grad */ false));
            }
            if (m_kahan_summation) {
                m_kahan_compensation.emplace(
                    key,
                    autograd::create_tensor(
                        core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                        /* requires_grad */ false));
            }
        }
    }
    init_param_groups(param_group_defaults(config), std::move(param_groups));
}

void AdamWComposite::zero_grad() {
    for (auto& [key, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            // setting gradients to not initialized tensor
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}

void AdamWComposite::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    m_steps++;
    for (const auto& group : m_param_groups) {
        const float lr = group.options.at("lr");
        const float beta1 = group.options.at("beta1");
        const float beta2 = group.options.at("beta2");
        const float epsilon = group.options.at("epsilon");
        const float weight_decay = group.options.at("weight_decay");

        for (const auto& key : group.params) {
            const auto first_moment_it = m_first_moment.find(key);
            if (first_moment_it == m_first_moment.end()) {
                continue;  // not trainable
            }
            auto& first_moment_ptr = first_moment_it->second;
            const auto& tensor_ptr = m_parameters.at(key);
            if (!tensor_ptr->is_grad_initialized()) {
                continue;
            }
            auto& second_moment_ptr = m_second_moment.at(key);
            auto first_moment = first_moment_ptr->get_value(autograd::PreferredPrecision::HALF);
            auto second_moment = second_moment_ptr->get_value(autograd::PreferredPrecision::HALF);

            auto gradients = tensor_ptr->get_grad();

            // By value: the ops below relabel their outputs (see optimizers::restore_topology).
            const auto topology = tensor_ptr->get_value(autograd::PreferredPrecision::HALF).tensor_topology();

            if (weight_decay != 0.0F) {
                auto weight_decay_update =
                    ttnn::multiply(tensor_ptr->get_value(autograd::PreferredPrecision::HALF), weight_decay * lr);
                // weights -= weight_decay * lr * weights
                tensor_ptr->set_value(
                    ttnn::subtract(tensor_ptr->get_value(autograd::PreferredPrecision::HALF), weight_decay_update));
            }

            // first moment = beta1 * first moment + (1 - beta1) * gradients
            first_moment = ttnn::add(ttnn::multiply(first_moment, beta1), ttnn::multiply(gradients, 1.F - beta1));
            // second moment = beta2 * second moment + (1 - beta2) * gradients^2
            second_moment =
                ttnn::add(ttnn::multiply(second_moment, beta2), ttnn::multiply(ttnn::square(gradients), 1.F - beta2));
            // first_moment_hat = first_moment / (1 - beta1^steps)
            auto first_moment_hat =
                ttnn::multiply(first_moment, static_cast<float>(1.F / (1.F - std::pow(beta1, m_steps))));

            first_moment_ptr->set_value(first_moment);
            second_moment_ptr->set_value(second_moment);

            // For amsgrad, use the maximum of all past second_moment values
            ttnn::Tensor denom_tensor;
            if (m_amsgrad) {
                auto& max_exp_avg_sq_ptr = m_max_exp_avg_sq.at(key);
                auto max_exp_avg_sq = max_exp_avg_sq_ptr->get_value(autograd::PreferredPrecision::HALF);
                // max_exp_avg_sq = max(max_exp_avg_sq, second_moment)
                max_exp_avg_sq = ttnn::maximum(max_exp_avg_sq, second_moment);
                max_exp_avg_sq_ptr->set_value(max_exp_avg_sq);
                // Apply bias correction after taking max
                auto max_exp_avg_sq_hat =
                    ttnn::multiply(max_exp_avg_sq, static_cast<float>(1.0f / (1.0f - std::pow(beta2, m_steps))));
                denom_tensor = ttnn::add(ttnn::sqrt(max_exp_avg_sq_hat), epsilon);
            } else {
                // second_moment_hat = second_moment / (1 - beta2^steps)
                auto second_moment_hat =
                    ttnn::multiply(second_moment, static_cast<float>(1.0f / (1.0f - std::pow(beta2, m_steps))));
                denom_tensor = ttnn::add(ttnn::sqrt(second_moment_hat), epsilon);
            }

            // weights -= lr * first_moment_hat / (sqrt(denom) + epsilon)
            auto update_tensor = ttnn_fixed::divide(ttnn::multiply(first_moment_hat, -lr), denom_tensor);

            if (!m_kahan_summation) {
                tensor_ptr->set_value(
                    ttnn::add(tensor_ptr->get_value(autograd::PreferredPrecision::HALF), update_tensor));
            } else {
                auto value_tensor = tensor_ptr->get_value(autograd::PreferredPrecision::HALF);

                const auto& kahan_compensation_ptr = m_kahan_compensation.at(key);
                // A running compensation for lost low-order bits
                auto compensation_tensor = kahan_compensation_ptr->get_value(autograd::PreferredPrecision::HALF);
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

            restore_topology({tensor_ptr, first_moment_ptr, second_moment_ptr}, topology);
            if (m_amsgrad) {
                restore_topology({m_max_exp_avg_sq.at(key)}, topology);
            }
            if (m_kahan_summation) {
                restore_topology({m_kahan_compensation.at(key)}, topology);
            }
        }
    }
}

[[nodiscard]] serialization::StateDict AdamWComposite::get_state_dict() const {
    serialization::StateDict state_dict;
    state_dict[kFirstMoment] = m_first_moment;
    state_dict[kSecondMoment] = m_second_moment;
    state_dict[kMaxExpAvgSq] = m_max_exp_avg_sq;
    state_dict[kKahanCompensation] = m_kahan_compensation;
    state_dict[kSteps] = m_steps;
    save_param_groups(state_dict);

    return state_dict;
}

void AdamWComposite::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_first_moment = std::get<serialization::NamedParameters>(dict.at(kFirstMoment));
    m_second_moment = std::get<serialization::NamedParameters>(dict.at(kSecondMoment));
    m_max_exp_avg_sq = std::get<serialization::NamedParameters>(dict.at(kMaxExpAvgSq));
    m_kahan_compensation = std::get<serialization::NamedParameters>(dict.at(kKahanCompensation));
    m_steps = serialization::get_value_type<size_t>(dict, kSteps);
}

[[nodiscard]] size_t AdamWComposite::get_steps() const {
    return m_steps;
}

void AdamWComposite::set_steps(size_t steps) {
    m_steps = steps;
}

}  // namespace ttml::optimizers
