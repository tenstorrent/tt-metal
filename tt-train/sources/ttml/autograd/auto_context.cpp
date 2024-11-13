// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>

namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() {
    return m_generator;
}

void AutoContext::set_seed(uint32_t seed) {
    m_seed = seed;
    m_generator = std::mt19937(m_seed);
}

uint32_t AutoContext::get_seed() const {
    return m_seed;
}

AutoContext& AutoContext::get_instance() {
    static AutoContext instance;
    return instance;
}
std::optional<NodeId> AutoContext::add_backward_node(GradFunction&& grad_function, std::span<NodeId> links) {
    if (m_grads_mode == GradMode::DISABLED) {
        return std::nullopt;
    }
    return m_graph.add_node(std::move(grad_function), links);
}
void AutoContext::set_gradient_mode(GradMode mode) {
    m_grads_mode = mode;
}
GradMode AutoContext::get_gradient_mode() const {
    return m_grads_mode;
}

void AutoContext::reset_graph() {
    m_graph.reset();
}

tt::tt_metal::Device& AutoContext::get_device() {
    return device.get_device();
}

AutoContext::AutoContext() : m_generator(m_seed) {
}
}  // namespace ttml::autograd
