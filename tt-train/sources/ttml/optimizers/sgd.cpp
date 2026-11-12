// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

SGD::SGD(
    ttml::serialization::NamedParameters parameters, const SGDConfig& config, std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)), m_nesterov(config.nesterov) {
    require_bf16_parameters(m_parameters, "SGD");
    init_param_groups(param_group_defaults(config), std::move(param_groups));
    validate_config();
    for (const auto& group : m_param_groups) {
        if (group.options.at("momentum") <= 0.0F) {
            continue;
        }
        for (const auto& name : group.params) {
            const auto& tensor_ptr = m_parameters.at(name);
            if (tensor_ptr->get_requires_grad()) {
                m_momentum.emplace(
                    name,
                    autograd::create_tensor(
                        core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                        /* requires_grad */ false));
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGD::step() {
    validate_config();

    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (const auto& group : m_param_groups) {
        const float lr = group.options.at("lr");
        const float momentum = group.options.at("momentum");
        const float group_dampening = group.options.at("dampening");
        const float weight_decay = group.options.at("weight_decay");

        for (const auto& name : group.params) {
            const auto& theta_ptr = m_parameters.at(name);
            if (!theta_ptr->is_grad_initialized()) {
                continue;
            }

            auto gradients = theta_ptr->get_grad();
            // The kernel updates the parameter and the momentum buffer in place, and only exists for bf16.
            auto param = theta_ptr->get_value_for_update(autograd::PreferredPrecision::HALF);

            std::optional<autograd::MutableTensorView> momentum_buffer;
            bool first_momentum_update = false;
            if (momentum > 0.0F) {
                auto it = m_momentum.find(name);
                if (it == m_momentum.end()) {
                    auto buf = autograd::create_tensor(
                        core::zeros_like(param.tensor()),
                        /* requires_grad */ false);
                    it = m_momentum.emplace(name, std::move(buf)).first;
                }
                first_momentum_update = m_momentum_initialized.insert(name).second;
                momentum_buffer.emplace(it->second->get_value_for_update(autograd::PreferredPrecision::HALF));
            }

            // A buffer's first update must produce buf = g (PyTorch seeds fresh buffers with the raw
            // gradient); the buffer is zero at that point, so skipping dampening is sufficient.
            const float dampening = first_momentum_update ? 0.0F : group_dampening;

            ttml::metal::sgd(
                param.tensor(),
                gradients,
                lr,
                momentum,
                dampening,
                weight_decay,
                m_nesterov,
                autograd::optional_tensor(momentum_buffer));
        }
    }
    m_steps++;
}

serialization::StateDict SGD::get_state_dict() const {
    serialization::StateDict dict;
    dict["steps"] = m_steps;
    dict["nesterov"] = m_nesterov;
    dict["momentum"] = m_momentum;
    save_param_groups(dict);
    return dict;
}

void SGD::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
    m_nesterov = serialization::get_value_type<bool>(dict, "nesterov");
    m_momentum = std::get<serialization::NamedParameters>(dict.at("momentum"));
    // Restored buffers are treated as past their first update. Buffers are preallocated
    // for every trainable parameter, so a buffer that never received a gradient before the
    // checkpoint (frozen parameter, or a checkpoint saved before any step) is also marked,
    // and its first resumed update applies dampening to a zero buffer — an accepted corner
    // case, since distinguishing it would require serializing this set.
    m_momentum_initialized.clear();
    for (const auto& [name, tensor_ptr] : m_momentum) {
        m_momentum_initialized.insert(name);
    }
}

size_t SGD::get_steps() const {
    return m_steps;
}

void SGD::set_steps(size_t steps) {
    this->m_steps = steps;
}

bool SGD::get_nesterov() const {
    return m_nesterov;
}

void SGD::set_nesterov(bool nesterov) {
    m_nesterov = nesterov;
}

void SGD::validate_config() const {
    for (size_t group_idx = 0; group_idx < m_param_groups.size(); ++group_idx) {
        const auto& options = m_param_groups[group_idx].options;
        const float momentum = options.at("momentum");
        const float dampening = options.at("dampening");
        if (m_nesterov) {
            if (dampening != 0.0F) {
                throw std::runtime_error(fmt::format(
                    "Nesterov momentum requires zero dampening! dampening={} (param group {})", dampening, group_idx));
            }
            if (momentum <= 0.0F) {
                throw std::runtime_error(fmt::format(
                    "Nesterov momentum requires a positive momentum! momentum={} (param group {})",
                    momentum,
                    group_idx));
            }
        }
        if (dampening != 0.0F && momentum == 0.0F) {
            throw std::runtime_error(fmt::format(
                "Dampening requires a positive momentum! momentum={} (param group {})", momentum, group_idx));
        }
    }
}

}  // namespace ttml::optimizers
