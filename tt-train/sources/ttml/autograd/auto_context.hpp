// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <random>

#include "core/device.hpp"
#include "graph.hpp"

namespace ttml::autograd {

enum class GradMode { ENABLED, DISABLED };

class AutoContext {
public:
    // Delete copy constructor and assignment operator to prevent copying
    AutoContext(const AutoContext&) = delete;
    AutoContext& operator=(const AutoContext&) = delete;
    AutoContext(AutoContext&&) = delete;
    AutoContext& operator=(AutoContext&&) = delete;
    // Static method to access the singleton instance
    static AutoContext& get_instance();

    std::mt19937& get_generator();

    void set_seed(uint32_t seed);

    [[nodiscard]] uint32_t get_seed() const;

    std::optional<NodeId> add_backward_node(GradFunction&& grad_function, std::span<NodeId> links);

    void reset_graph();

    void set_gradient_mode(GradMode mode);

    [[nodiscard]] GradMode get_gradient_mode() const;

    ~AutoContext() = default;  // to make it work with unique_ptr.

    tt::tt_metal::Device& get_device();

private:
    AutoContext();
    uint32_t m_seed = 5489U;
    std::mt19937 m_generator;

    GradMode m_grads_mode = GradMode::ENABLED;

    Graph m_graph;

    core::Device device{0};
};

inline auto& ctx() {
    return AutoContext::get_instance();
}
}  // namespace ttml::autograd
