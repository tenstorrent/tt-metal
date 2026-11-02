// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw.hpp"

#include <fmt/format.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <ttnn/operations/data_movement/copy/copy.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
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
    const char* device_step = std::getenv("TTML_ADAMW_DEVICE_STEP");
    m_device_step = device_step != nullptr && std::string(device_step) == "1";
    if (m_device_step) {
        reset_beta_pow_tensors();
    }
}

void AdamW::reset_beta_pow_tensors() {
    auto* device = &autograd::ctx().get_device();
    m_beta_pow_tensors.clear();
    m_beta_tensors.clear();
    m_tile0_masks.clear();
    m_tile1_masks.clear();
    for (const auto& group : m_param_groups) {
        const float beta1 = group.options.at("beta1");
        const float beta2 = group.options.at("beta2");
        const auto pow1 = static_cast<float>(std::pow(beta1, m_steps));
        const auto pow2 = static_cast<float>(std::pow(beta2, m_steps));
        const ttnn::Shape tile_shape({1, 1, 32, 32});
        auto two_tiles = [&](float a, float b) {
            return ttnn::concat(
                std::vector<ttnn::Tensor>{
                    core::full(tile_shape, a, device, ttnn::DataType::FLOAT32),
                    core::full(tile_shape, b, device, ttnn::DataType::FLOAT32)},
                3);
        };
        m_beta_pow_tensors.push_back(two_tiles(pow1, pow2));
        m_beta_tensors.push_back(two_tiles(beta1, beta2));
        m_tile0_masks.push_back(two_tiles(1.0F, 0.0F));
        m_tile1_masks.push_back(two_tiles(0.0F, 1.0F));
    }
}

ttnn::Tensor AdamW::advance_step_scalars_on_device(size_t group_idx, float lr) {
    // pows *= betas in place (the buffer is the cross-step state a trace replays on), then the scalars:
    // tile 0 = (1 - beta1^t) / lr, tile 1 = 1 / sqrt(1 - beta2^t).
    auto& pows = m_beta_pow_tensors[group_idx];
    auto next = ttnn::multiply(pows, m_beta_tensors[group_idx]);
    ttnn::copy(next, pows);
    auto bc = ttnn::multiply(ttnn::add(pows, -1.0F), -1.0F);  // 1 - pow, both tiles
    // tile 0 keeps (1 - beta1^t) / lr, tile 1 1 / sqrt(1 - beta2^t): the two formulas over both tiles, masked
    auto inv_step = ttnn::multiply(ttnn::multiply(bc, 1.0F / lr), m_tile0_masks[group_idx]);
    auto inv_sqrt = ttnn::multiply(ttnn::rsqrt(bc), m_tile1_masks[group_idx]);
    return ttnn::add(inv_step, inv_sqrt);
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
        std::optional<ttnn::Tensor> step_scalars;
        if (m_device_step) {
            step_scalars = advance_step_scalars_on_device(group_idx, lr);
        }

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
                                    : std::nullopt,
                step_scalars);
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
    if (m_device_step) {
        reset_beta_pow_tensors();
    }
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
