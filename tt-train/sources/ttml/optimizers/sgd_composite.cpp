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

SGDComposite::SGDComposite(ttml::serialization::NamedParameters parameters, const SGDCompositeConfig& config) :
    OptimizerBase(std::move(parameters)), m_config(config) {
    for (const auto& [name, tensor_ptr] : m_parameters) {
        if (tensor_ptr->get_requires_grad()) {
            m_theta.emplace(
                name,
                autograd::create_tensor(
                    core::zeros_like(tensor_ptr->get_value(autograd::PreferredPrecision::HALF)),
                    /* requires_grad */ false));
        }
    }
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

    for (auto& [name, theta_ptr] : m_theta) {
        auto theta = theta_ptr->get_value(autograd::PreferredPrecision::HALF);
        const auto& tensor_ptr = m_parameters.at(name);
        if (!tensor_ptr->is_grad_initialized()) {
            continue;
        }

        auto gradients = tensor_ptr->get_grad();

        if (m_config.weight_decay != 0.0F) {
            gradients = ttnn::add(
                ttnn::multiply(
                    tensor_ptr->get_value(autograd::PreferredPrecision::HALF),
                    m_config.weight_decay,
                    /* fast_and_approximate_mode*/ true),
                gradients);
        }

        if (m_config.momentum != 0.0F) {
            // A buffer's first update must produce theta = g (PyTorch seeds fresh buffers with
            // the raw gradient); the buffer is zero at that point, so momentum and dampening
            // are skipped for it.
            const bool first_momentum_update = m_theta_initialized.insert(name).second;
            if (!first_momentum_update) {
                theta = ttnn::multiply(
                    theta,
                    m_config.momentum,
                    /* fast_and_approximate_mode*/ true);
            }
            const float dampening = first_momentum_update ? 0.0F : m_config.dampening;
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

            if (m_config.nesterov) {
                gradients = ttnn::add(
                    gradients,
                    ttnn::multiply(
                        theta,
                        m_config.momentum,
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
                m_config.lr,
                /* fast_and_approximate_mode*/ true)));
    }
    m_steps++;
}

serialization::StateDict SGDComposite::get_state_dict() const {
    serialization::StateDict dict;
    dict["theta"] = m_theta;
    dict["steps"] = m_steps;
    dict["lr"] = m_config.lr;
    return dict;
}

void SGDComposite::set_state_dict(const serialization::StateDict& dict) {
    m_theta = std::get<serialization::NamedParameters>(dict.at("theta"));
    m_steps = serialization::get_value_type<size_t>(dict, "steps");
    set_lr(serialization::get_value_type<float>(dict, "lr"));
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
