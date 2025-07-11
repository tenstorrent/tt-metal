// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>

namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() {
    return m_generator;
}

void AutoContext::set_generator(const std::mt19937& generator) {
    m_generator = generator;
}

void AutoContext::set_seed(uint32_t seed) {
    m_seed = seed;
    m_generator = std::mt19937(m_seed);
}

uint32_t AutoContext::get_seed() const {
    return m_seed;
}

AutoContext& AutoContext::get_instance() {
    static ttsl::Indestructible<AutoContext> instance{};
    return instance.get();
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

void AutoContext::open_device(
    const tt::tt_metal::distributed::MeshShape& mesh_shape, const std::vector<int>& device_ids) {
    if (m_device) {
        return;
        // throw std::runtime_error("open_device was called after the device was created.");
    }
    m_mesh_shape = mesh_shape;
    std::cout << "Creating MeshDevice with shape: " << m_mesh_shape << std::endl,
        m_device = std::make_unique<core::MeshDevice>(m_mesh_shape, device_ids);
}

void AutoContext::close_device() {
    m_device = nullptr;
}

ttnn::distributed::MeshDevice& AutoContext::get_device() {
    if (!m_device) {
        open_device();
    }

    return m_device->get_device();
}

std::shared_ptr<ttnn::distributed::MeshDevice> AutoContext::get_shared_ptr_device() {
    if (!m_device) {
        open_device();
    }
    return m_device->get_shared_ptr_device();
}

AutoContext::AutoContext() : m_generator(m_seed) {
}

tt::tt_metal::distributed::MeshShape AutoContext::get_mesh_shape() const {
    return m_mesh_shape;
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> AutoContext::get_distributed_context() const {
    if (!m_distributed_context) {
        throw std::runtime_error("DistributedContext is not initialized.");
    }
    return m_distributed_context;
}

void AutoContext::initialize_distributed_context(int argc, char** argv) {
    if (m_distributed_context) {
        throw std::runtime_error("MPIContext is already initialized.");
    }
    // tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    // m_distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    m_distributed_context = this->get_shared_ptr_device()->get_distributed_context();
}

void AutoContext::set_fabric_config(
    const std::string& mesh_graph_descriptor_path,
    const std::vector<std::vector<std::vector<uint32_t>>>& eth_coords_per_mesh) {
    this->get_shared_ptr_device()->initialize_control_plane_config(mesh_graph_descriptor_path, eth_coords_per_mesh);
}

}  // namespace ttml::autograd
