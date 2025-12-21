// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "auto_context.hpp"
#include "core/template_utils.hpp"
#include "graph.hpp"

namespace ttml::autograd {

template <typename... Tensors>
std::vector<NodeId> get_links(Tensors&&... tensors) {
    static_assert(core::are_same_type<Tensors...>(), "All nodes must have the same type!");

    std::vector<NodeId> links;
    links.reserve(sizeof...(Tensors));
    auto process_node = [&links](auto&& tensor) {
        if (tensor == nullptr) {
            return;
        }

        const auto& node = tensor->get_node();
        if (node) {
            links.push_back(node.value());
        }
    };

    (process_node(std::forward<Tensors>(tensors)), ...);

    return links;
}

template <typename... Tensors>
bool any_requires_grad(Tensors&&... tensors) {
    static_assert(core::are_same_type<Tensors...>(), "All nodes must have the same type!");

    bool requires_grad = false;
    auto check_tensor = [&requires_grad](auto&& tensor) {
        if (tensor != nullptr && tensor->get_requires_grad()) {
            requires_grad = true;
        }
    };

    (check_tensor(std::forward<Tensors>(tensors)), ...);

    return requires_grad;
}

template <typename... Tensors>
std::optional<NodeId> add_backward_node(GradFunction&& grad_function, const TensorPtr& output, Tensors&&... tensors) {
    static_assert(core::are_same_type<Tensors...>(), "All nodes must have the same type!");

    // Check if any input tensor requires gradients
    bool needs_grad = any_requires_grad(std::forward<Tensors>(tensors)...);

    // Set output tensor's requires_grad based on whether any input needs gradients
    output->set_requires_grad(needs_grad);

    if (!needs_grad) {
        return std::nullopt;
    }

    // Get links and add node normally
    auto links = get_links(std::forward<Tensors>(tensors)...);
    return ctx().add_backward_node(std::move(grad_function), links);
}

}  // namespace ttml::autograd
