// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>

#include "core/tt_profiler.hpp"
#include "core/tt_tensor_utils.hpp"

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

[[nodiscard]] ParallelismContext& AutoContext::get_parallelism_context() {
    return m_parallelism_context;
}

void ParallelismContext::configure(
    ttnn::distributed::MeshDevice* mesh_device, bool enable_dp, bool enable_tp, bool enable_cp) {
    TT_FATAL(
        !(enable_dp && enable_cp),
        "DP and CP cannot be enabled simultaneously. This combination is not currently supported.");

    TT_FATAL(
        (uint32_t)enable_dp + (uint32_t)enable_cp + (uint32_t)enable_tp <= mesh_device->shape().dims(),
        "Mesh shape dimensions must be greater than the number of parallelization axes");

    m_mesh_device = mesh_device;

    // Axis assignment order: DP -> CP -> TP
    uint32_t axis = 0;
    if (enable_dp) {
        m_dp_axis = axis++;
    }
    if (enable_cp) {
        m_cp_axis = axis++;
    }
    if (enable_tp) {
        m_tp_axis = axis++;
    }

    // Create CP rank tensor if CP is enabled
    if (enable_cp) {
        create_cp_rank_tensor();
    }
}

void ParallelismContext::create_cp_rank_tensor() {
    if (!m_cp_axis.has_value() || m_mesh_device == nullptr) {
        return;
    }

    uint32_t cp_size = get_cp_size();

    // Create rank values [0, 1, 2, ..., cp_size-1] as float
    // Shape: (1, 1, 1, cp_size) - will be sharded along last dim so each device gets one value
    std::vector<float> rank_values(cp_size);
    for (uint32_t i = 0; i < cp_size; ++i) {
        rank_values[i] = static_cast<float>(i);
    }

    // Shard along dim 3 (last dim) on CP axis - each device gets its rank
    auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*m_mesh_device, /*dim=*/3, m_cp_axis);
    m_cp_rank_tensor = core::from_vector(
        rank_values, ttnn::Shape({1U, 1U, 1U, cp_size}), m_mesh_device, ttnn::Layout::ROW_MAJOR, mapper.get());
}

std::optional<ttnn::Tensor> ParallelismContext::get_cp_rank_tensor() const {
    return m_cp_rank_tensor;
}

uint32_t ParallelismContext::get_dp_size() const {
    if (!m_dp_axis.has_value() || m_mesh_device == nullptr) {
        return 1U;
    }
    return m_mesh_device->shape()[m_dp_axis.value()];
}

uint32_t ParallelismContext::get_cp_size() const {
    if (!m_cp_axis.has_value() || m_mesh_device == nullptr) {
        return 1U;
    }
    return m_mesh_device->shape()[m_cp_axis.value()];
}

uint32_t ParallelismContext::get_tp_size() const {
    if (m_mesh_device == nullptr) {
        return 1U;
    }
    if (!m_tp_axis.has_value()) {
        return static_cast<uint32_t>(m_mesh_device->num_devices());
    }
    return m_mesh_device->shape()[m_tp_axis.value()];
}

}  // namespace ttml::autograd
