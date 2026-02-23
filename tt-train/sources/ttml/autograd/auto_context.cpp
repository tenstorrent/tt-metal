// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>

#include "core/tt_profiler.hpp"

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
        throw std::runtime_error("open_device was called after the device was created.");
    }
    m_mesh_shape = mesh_shape;
    m_device = std::make_unique<core::MeshDevice>(m_mesh_shape, device_ids);
}

void AutoContext::close_profiler() {
    m_profiler = nullptr;
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

[[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> AutoContext::get_device_ptr() {
    if (!m_device) {
        open_device();
    }

    return m_device->get_device_ptr();
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

    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    m_distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
}

core::TTProfiler& AutoContext::get_profiler() {
    if (!m_profiler) {
        m_profiler = std::make_unique<core::TTProfiler>();
    }
    return *m_profiler;
}

[[nodiscard]] core::distributed::CCLResources& AutoContext::get_ccl_resources() {
    if (!m_ccl_resources) {
        m_ccl_resources = std::make_unique<core::distributed::CCLResources>();
    }
    return *m_ccl_resources;
}

void AutoContext::initialize_socket_manager(ttnn::distributed::SocketType socket_type) {
    m_socket_manager = std::make_unique<core::distributed::SocketManager>(socket_type);
}

[[nodiscard]] core::distributed::SocketManager& AutoContext::get_socket_manager() {
    if (!m_socket_manager) {
        throw std::runtime_error("SocketManager is not initialized. Do not forget to call initialize_socket_manager.");
    }
    return *m_socket_manager;
}

ParallelismContext::ParallelismContext(
    const ttnn::distributed::MeshDevice& mesh_device, const DistributedConfig& config) {
    const uint32_t num_enabled_parallelisms =
        (uint32_t)config.enable_ddp + (uint32_t)config.enable_cp + (uint32_t)config.enable_tp;
    const auto& mesh_shape = mesh_device.shape();

    // Check if this is a line topology (one dimension is 1, e.g., [1, 32] or [32, 1])
    // For line topologies, only one parallelism type can be enabled
    const bool is_line_topology = mesh_shape.is_line_topology();

    if (is_line_topology) {
        TT_FATAL(
            num_enabled_parallelisms == 1,
            "For line mesh topology (shape {}), exactly one parallelism type must be enabled. "
            "Got: ddp={}, tp={}, cp={}",
            mesh_shape,
            config.enable_ddp,
            config.enable_tp,
            config.enable_cp);

        // Find the non-trivial axis (the one with size > 1)
        uint32_t active_axis = 0;
        for (uint32_t i = 0; i < mesh_shape.dims(); ++i) {
            if (mesh_shape[i] > 1) {
                active_axis = i;
                break;
            }
        }

        // Assign the single enabled parallelism to the active axis
        if (config.enable_ddp) {
            m_ddp_axis = active_axis;
            m_num_ddp_devices = mesh_shape[active_axis];
        } else if (config.enable_cp) {
            m_cp_axis = active_axis;
            m_num_cp_devices = mesh_shape[active_axis];
        } else if (config.enable_tp) {
            m_tp_axis = active_axis;
            m_num_tp_devices = mesh_shape[active_axis];
        }
    } else {
        // For 2D meshes (both dimensions > 1), number of parallelisms must match mesh dimensions
        TT_FATAL(
            num_enabled_parallelisms == mesh_shape.dims(),
            "For 2D mesh (shape {}), number of enabled parallelization axes ({}) must equal mesh dimensions ({}).",
            mesh_shape,
            num_enabled_parallelisms,
            mesh_shape.dims());

        // Axis assignment order: DP -> CP -> TP
        uint32_t axis = 0;
        if (config.enable_ddp) {
            m_ddp_axis = axis++;
            m_num_ddp_devices = mesh_shape[m_ddp_axis.value()];
        }
        if (config.enable_cp) {
            m_cp_axis = axis++;
            m_num_cp_devices = mesh_shape[m_cp_axis.value()];
        }
        if (config.enable_tp) {
            m_tp_axis = axis++;
            m_num_tp_devices = mesh_shape[m_tp_axis.value()];
        }
    }
}

[[nodiscard]] const ParallelismContext& AutoContext::get_parallelism_context() const {
    if (!m_parallelism_context) {
        throw std::runtime_error("ParallelismContext is not initialized.");
    }
    return *m_parallelism_context;
}

bool AutoContext::is_parallelism_context_initialized() const {
    return m_parallelism_context != nullptr;
}

void AutoContext::initialize_parallelism_context(const DistributedConfig& config) {
    if (m_parallelism_context) {
        throw std::runtime_error("ParallelismContext is already initialized.");
    }
    m_parallelism_context = std::make_unique<ParallelismContext>(get_device(), config);
}

const uint32_t ParallelismContext::get_ddp_size() const {
    if (!m_ddp_axis.has_value()) {
        return 1U;
    }
    return m_num_ddp_devices;
}

const uint32_t ParallelismContext::get_cp_size() const {
    if (!m_cp_axis.has_value()) {
        return 1U;
    }
    return m_num_cp_devices;
}

const uint32_t ParallelismContext::get_tp_size() const {
    if (!m_tp_axis.has_value()) {
        return 1U;
    }
    return m_num_tp_devices;
}

}  // namespace ttml::autograd
