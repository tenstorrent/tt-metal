// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_composite.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/debug.hpp"
#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

SGDComposite::SGDComposite(
    ttml::serialization::NamedParameters parameters,
    const SGDCompositeConfig& config,
    std::vector<ParamGroup> param_groups) :
    OptimizerBase(std::move(parameters)), m_nesterov(config.nesterov) {
    require_bf16_parameters(m_parameters, "SGDComposite");
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_theta.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
        }
    }
    init_param_groups(param_group_defaults(config), std::move(param_groups));
}

void SGDComposite::zero_grad() {
    for (auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad() && tensor_ptr->is_grad_initialized()) {
            tensor_ptr->set_grad(core::zeros_like(tensor_ptr->get_value()));
        }
    }
}

void SGDComposite::step() {
    if (core::debug::Debug::enable_print_tensor_stats()) {
        print_stats();
    }

    for (const auto& group : m_param_groups) {
        const float lr = group.options.at("lr");
        const float momentum = group.options.at("momentum");
        const float group_dampening = group.options.at("dampening");
        const float weight_decay = group.options.at("weight_decay");

        for (const auto& name : group.params) {
            const auto theta_it = m_theta.find(name);
            if (theta_it == m_theta.end()) {
                continue;
            }
            auto& theta_ptr = theta_it->second;
            auto theta = theta_ptr->get_value(autograd::PreferredPrecision::HALF);
            const auto& tensor_ptr = m_parameters.at(name);
            if (!tensor_ptr->is_grad_initialized()) {
                continue;
            }

            auto gradients = tensor_ptr->get_grad();

            // By value: the ops below relabel their outputs (see optimizers::restore_topology).
            const auto topology = tensor_ptr->get_value(autograd::PreferredPrecision::HALF).tensor_topology();

            if (weight_decay != 0.0F) {
                gradients = ttnn::add(
                    ttnn::multiply(
                        tensor_ptr->get_value(autograd::PreferredPrecision::HALF),
                        weight_decay,
                        /* fast_and_approximate_mode*/ true),
                    gradients);
            }

            if (momentum != 0.0F) {
                // A buffer's first update must produce theta = g (PyTorch seeds fresh buffers with
                // the raw gradient); the buffer is zero at that point, so momentum and dampening
                // are skipped for it.
                const bool first_momentum_update = m_theta_initialized.insert(name).second;
                if (!first_momentum_update) {
                    theta = ttnn::multiply(
                        theta,
                        momentum,
                        /* fast_and_approximate_mode*/ true);
                }
                const float dampening = first_momentum_update ? 0.0F : group_dampening;
                if (dampening != 0.0F) {
                    theta = ttnn::add(
                        theta,
                        ttnn::multiply(
                            gradients,
                            1 - dampening,
                            /* fast_and_approximate_mode*/ true));
                } else {
                    theta = ttnn::add(theta, gradients);
                }

                if (m_nesterov) {
                    gradients = ttnn::add(
                        gradients,
                        ttnn::multiply(
                            theta,
                            momentum,
                            /* fast_and_approximate_mode*/ true));
                } else {
                    gradients = theta;
                }
            }
            theta_ptr->set_value(theta);
            tensor_ptr->set_value(ttnn::subtract(
                tensor_ptr->get_value(autograd::PreferredPrecision::HALF),
                ttnn::multiply(
                    gradients,
                    lr,
                    /* fast_and_approximate_mode*/ true)));
            restore_topology({tensor_ptr, theta_ptr}, topology);
        }
    }
    m_steps++;
}

serialization::StateDict SGDComposite::get_state_dict() const {
    serialization::StateDict dict;
    dict["theta"] = m_theta;
    dict["steps"] = m_steps;
    save_param_groups(dict);
    return dict;
}

void SGDComposite::set_state_dict(const serialization::StateDict& dict) {
    restore_param_groups(dict);
    m_theta = std::get<serialization::NamedParameters>(dict.at("theta"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
    // Restored buffers are treated as past their first update. Buffers are preallocated
    // for every trainable parameter, so a buffer that never received a gradient before the
    // checkpoint (frozen parameter, or a checkpoint saved before any step) is also marked,
    // and its first resumed update applies dampening to a zero buffer — an accepted corner
    // case, since distinguishing it would require serializing this set.
    m_theta_initialized.clear();
    for (const auto& [name, tensor_ptr] : m_theta) {
        m_theta_initialized.insert(name);
    }
}

size_t SGDComposite::get_steps() const {
    return m_steps;
}

void SGDComposite::set_steps(size_t steps) {
    this->m_steps = steps;
}

}  // namespace ttml::optimizers
