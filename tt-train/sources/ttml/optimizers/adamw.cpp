// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

namespace {
// A zero optimizer-state tensor for a parameter. The kernel needs the state in the parameter's own dtype.
autograd::TensorPtr zeros_state_like(const autograd::TensorPtr& parameter) {
    return autograd::create_tensor(
        core::zeros_like(parameter->get_value(autograd::PreferredPrecision::NATIVE)), /* requires_grad */ false);
}
}  // namespace

std::string AdamW::get_name() const {
    return "AdamW";
}

AdamW::AdamW(
    ttml::serialization::NamedParameters parameters, const AdamWConfig& config, std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)), m_amsgrad(config.amsgrad), m_stochastic_rounding(config.stochastic_rounding) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_exp_avg.emplace(name, zeros_state_like(tensor_ptr));
            m_exp_avg_sq.emplace(name, zeros_state_like(tensor_ptr));
        }
    }
    if (m_amsgrad) {
        init_max_exp_avg_sq();
    }

    init_param_groups(param_group_defaults(config), std::move(param_groups));
    m_beta_pows.reset(m_param_groups, m_steps);
}

void AdamW::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(ttnn::Tensor());
        }
    }
}

void AdamW::step() {
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
            // The kernel updates the parameter and its state in place: write through the native tensors.
            auto param = theta_ptr->get_value_for_update();
            auto exp_avg = m_exp_avg.at(name)->get_value_for_update();
            auto exp_avg_sq = m_exp_avg_sq.at(name)->get_value_for_update();

            std::optional<autograd::MutableTensorView> max_exp_avg_sq;
            if (m_amsgrad) {
                max_exp_avg_sq.emplace(m_max_exp_avg_sq.at(name)->get_value_for_update());
            }

            // Stochastic rounding recovers updates that bf16 storage would lose; an fp32 parameter doesn't lose them,
            // and the kernel supports it for bf16 parameters only.
            const bool stochastic_rounding =
                m_stochastic_rounding && param.tensor().dtype() == ttnn::DataType::BFLOAT16;

            ttml::metal::adamw(
                param.tensor(),
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
                weight_decay,
                static_cast<ttml::metal::StochasticRounding>(stochastic_rounding),
                stochastic_rounding ? std::optional<uint32_t>{static_cast<uint32_t>(autograd::ctx().get_generator()())}
                                    : std::nullopt);
        }
    }
}

serialization::StateDict AdamW::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    dict["amsgrad"] = m_amsgrad;
    dict["stochastic_rounding"] = m_stochastic_rounding;
    dict["exp_avg"] = m_exp_avg;
    dict["exp_avg_sq"] = m_exp_avg_sq;
    if (m_amsgrad) {
        dict["max_exp_avg_sq"] = m_max_exp_avg_sq;
    }
    save_param_groups(dict);
    return dict;
}

void AdamW::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_stochastic_rounding = serialization::get_value_type<bool>(dict, "stochastic_rounding");
    set_steps(serialization::get_value_type<size_t>(dict, "steps"));  // rebuilds the beta powers with pow
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

size_t AdamW::get_steps() const {
    return m_steps;
}

void AdamW::set_steps(size_t steps) {
    m_steps = steps;
    m_beta_pows.reset(m_param_groups, m_steps);
}

void AdamW::set_param_group_option(size_t group_idx, const std::string& key, float value) {
    OptimizerBase::set_param_group_option(group_idx, key, value);
    if (BetaPowers::depends_on(key)) {
        m_beta_pows.reset(m_param_groups, m_steps);
    }
}

bool AdamW::get_amsgrad() const {
    return m_amsgrad;
}

void AdamW::set_amsgrad(bool amsgrad) {
    if (m_amsgrad == amsgrad) {
        return;
    }
    m_amsgrad = amsgrad;
    amsgrad ? init_max_exp_avg_sq() : m_max_exp_avg_sq.clear();
}

bool AdamW::get_stochastic_rounding() const {
    return m_stochastic_rounding;
}

void AdamW::set_stochastic_rounding(bool stochastic_rounding) {
    m_stochastic_rounding = stochastic_rounding;
}

void AdamW::init_max_exp_avg_sq() {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_max_exp_avg_sq.emplace(name, zeros_state_like(tensor_ptr));
        }
    }
}
}  // namespace ttml::optimizers
