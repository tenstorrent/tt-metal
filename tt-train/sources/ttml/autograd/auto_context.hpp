// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <random>

#include "core/indestructible.hpp"
#include "core/mesh_device.hpp"
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
    void set_generator(const std::mt19937& generator);

    void set_seed(uint32_t seed);

    [[nodiscard]] uint32_t get_seed() const;

    std::optional<NodeId> add_backward_node(GradFunction&& grad_function, std::span<NodeId> links);

    void reset_graph();

    void set_gradient_mode(GradMode mode);

    [[nodiscard]] GradMode get_gradient_mode() const;

    ~AutoContext() = default;  // to make it work with unique_ptr.

    ttnn::distributed::MeshDevice& get_device();

    void set_mesh_shape(tt::tt_metal::distributed::MeshShape shape);
    [[nodiscard]] tt::tt_metal::distributed::MeshShape get_mesh_shape() const;

    void open_device();

    void close_device();

private:
    AutoContext();
    uint32_t m_seed = 5489U;
    std::mt19937 m_generator;

    GradMode m_grads_mode = GradMode::ENABLED;

    Graph m_graph;
    tt::tt_metal::distributed::MeshShape m_mesh_shape = {1, 1};
    std::unique_ptr<core::MeshDevice> m_device;

    friend class core::Indestructible<AutoContext>;
};

inline auto& ctx() {
    return AutoContext::get_instance();
}
}  // namespace ttml::autograd
