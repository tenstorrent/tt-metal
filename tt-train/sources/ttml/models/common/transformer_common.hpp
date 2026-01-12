// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/module_base.hpp"

namespace ttml::models::common::transformer {

enum class RunnerType {
    MemoryEfficient,
    Default,
};

enum class WeightTyingType {
    Disabled,
    Enabled,
};

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl, const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return forward_impl(input, mask);
    }

    // make a copy of a generator before running forward pass
    auto generator = autograd::ctx().get_generator();

    // running forward pass
    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = forward_impl(input, mask);
    }

    // define grad function and copy generator (in the state before forward pass)
    autograd::GradFunction grad = [input, mask, out, &forward_impl, generator]() {
        // detach input from existing graph
        auto input_detached = autograd::create_tensor(input->get_value());
        // enable gradients for the detached input so the graph is built during recomputation
        input_detached->set_requires_grad(true);
        // run forward pass again
        autograd::TensorPtr output;
        {
            // set generator to the state before forward pass during construction
            // restore generator state after grad function is executed
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = forward_impl(input_detached, mask);
        }
        // use gradients from new output
        output->set_grad(out->get_grad());
        output->backward();
        // reuse gradients from detached input
        input->add_grad(input_detached->get_grad());
    };

    // Add backward node unconditionally - we bypass add_backward_node's requires_grad check
    // because during recomputation, parameters might be trainable even if input isn't.
    // This is critical for LoRA where input from frozen embeddings has requires_grad=false
    // but internal LoRA parameters are trainable.
    auto links = autograd::get_links(input);
    auto node_id = autograd::ctx().add_backward_node(std::move(grad), links);
    out->set_node(node_id);
    // Set output requires_grad to true since we added a backward node
    out->set_requires_grad(true);
    return out;
}

void initialize_weights_gpt2(modules::ModuleBase& model);
void initialize_weights_he_kaiming_normal(modules::ModuleBase& model);

RunnerType read_runner_type(const YAML::Node& config);
WeightTyingType read_weight_tying_type(const YAML::Node& config);

}  // namespace ttml::models::common::transformer
