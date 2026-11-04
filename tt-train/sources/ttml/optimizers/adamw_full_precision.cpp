// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    ttml::serialization::NamedParameters parameters,
    const AdamWFullPrecisionConfig& config,
    std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)), m_amsgrad(config.amsgrad) {
    // The model parameters are the bf16 compute copies; the fp32 masters live in this optimizer.
    require_bf16_parameters(m_parameters, "AdamWFullPrecision");
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            // Create fp32 master weights from the initial bf16 weights
            auto bf16_weights = tensor_ptr->get_value(autograd::PreferredPrecision::HALF);
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
    if (m_amsgrad) {
        init_max_exp_avg_sq();
    }

    init_param_groups(param_group_defaults(config), std::move(param_groups));
    m_beta_pows.reset(m_param_groups, m_steps);
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

    for (size_t group_idx = 0; group_idx < m_param_groups.size(); ++group_idx) {
        const auto& group = m_param_groups[group_idx];
        const float lr = group.options.at("lr");
        const float beta1 = group.options.at("beta1");
        const float beta2 = group.options.at("beta2");
        const float epsilon = group.options.at("epsilon");
        const float weight_decay = group.options.at("weight_decay");
        const auto [beta1_pow, beta2_pow] = m_beta_pows.advance(group_idx, beta1, beta2);

        for (const auto& name : group.params) {
            const auto& theta_ptr = m_parameters.at(name);
            if (!theta_ptr->is_grad_initialized()) {
                continue;
            }

            auto gradients = theta_ptr->get_grad();

            // The fp32 master weights and moments are updated in place: write through their native tensors.
            auto master_weights = m_master_weights.at(name)->get_value_for_update(autograd::PreferredPrecision::FULL);
            auto exp_avg = m_exp_avg.at(name)->get_value_for_update(autograd::PreferredPrecision::FULL);
            auto exp_avg_sq = m_exp_avg_sq.at(name)->get_value_for_update(autograd::PreferredPrecision::FULL);

            std::optional<autograd::MutableTensorView> max_exp_avg_sq;
            if (m_amsgrad) {
                max_exp_avg_sq.emplace(
                    m_max_exp_avg_sq.at(name)->get_value_for_update(autograd::PreferredPrecision::FULL));
            }

            ttml::metal::adamw(
                master_weights.tensor(),
                gradients,
                exp_avg.tensor(),
                exp_avg_sq.tensor(),
                autograd::optional_tensor(max_exp_avg_sq),
                lr,
                beta1,
                beta2,
                beta1_pow,
                beta2_pow,
                epsilon,
                weight_decay);

            // Convert updated fp32 master weights back to bf16 visible weights
            auto updated_bf16_weights = ttnn::typecast(master_weights.tensor(), tt::tt_metal::DataType::BFLOAT16);
            theta_ptr->set_value(updated_bf16_weights);
        }
    }
}

serialization::StateDict AdamWFullPrecision::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    dict["amsgrad"] = m_amsgrad;
    dict["master_weights"] = m_master_weights;
    dict["exp_avg"] = m_exp_avg;
    dict["exp_avg_sq"] = m_exp_avg_sq;
    if (m_amsgrad) {
        dict["max_exp_avg_sq"] = m_max_exp_avg_sq;
    }
    save_param_groups(dict);
    return dict;
}

void AdamWFullPrecision::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    set_steps(serialization::get_value_type<size_t>(dict, "steps"));  // rebuilds the beta powers with pow
    m_master_weights = std::get<serialization::NamedParameters>(dict.at("master_weights"));
    m_exp_avg = std::get<serialization::NamedParameters>(dict.at("exp_avg"));
    m_exp_avg_sq = std::get<serialization::NamedParameters>(dict.at("exp_avg_sq"));

    const bool amsgrad = serialization::get_value_type<bool>(dict, "amsgrad");
    if (amsgrad && dict.contains("max_exp_avg_sq")) {
        m_amsgrad = true;
        m_max_exp_avg_sq = std::get<serialization::NamedParameters>(dict.at("max_exp_avg_sq"));
    } else {
        set_amsgrad(amsgrad);
    }
}

size_t AdamWFullPrecision::get_steps() const {
    return m_steps;
}

void AdamWFullPrecision::set_steps(size_t steps) {
    m_steps = steps;
    m_beta_pows.reset(m_param_groups, m_steps);
}

void AdamWFullPrecision::set_param_group_option(size_t group_idx, const std::string& key, float value) {
    OptimizerBase::set_param_group_option(group_idx, key, value);
    if (BetaPowers::depends_on(key)) {
        m_beta_pows.reset(m_param_groups, m_steps);
    }
}

bool AdamWFullPrecision::get_amsgrad() const {
    return m_amsgrad;
}

void AdamWFullPrecision::set_amsgrad(bool amsgrad) {
    if (m_amsgrad == amsgrad) {
        return;
    }
    m_amsgrad = amsgrad;
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
