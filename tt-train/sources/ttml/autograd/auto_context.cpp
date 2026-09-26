// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "auto_context.hpp"

#include <optional>
#include <sstream>

#include "core/tt_profiler.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() {
    return m_generator;
}

void AutoContext::set_generator(const std::mt19937& generator) {
    m_generator = generator;
}

std::string AutoContext::get_generator_state() const {
    std::ostringstream oss;
    oss << m_generator;  // mt19937's stream operator emits its full internal state (624 words + position)
    return oss.str();
}

void AutoContext::set_generator_state(const std::string& state) {
    std::istringstream iss(state);
    iss >> m_generator;
    if (iss.fail()) {
        throw std::runtime_error("Failed to deserialize RNG generator state.");
    }
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

void AutoContext::enter_backward() {
    ++m_backward_depth;
}

void AutoContext::exit_backward() {
    TT_FATAL(m_backward_depth > 0U, "AutoContext::exit_backward called without a matching enter_backward");
    --m_backward_depth;
}

bool AutoContext::is_backward_in_progress() const {
    return m_backward_depth > 0U;
}

void AutoContext::enable_ccl_sub_device(uint32_t num_columns, uint32_t num_rows) {
    TT_FATAL(!m_ccl_sub_device_id.has_value(), "AutoContext: the CCL sub-device is already enabled");
    TT_FATAL(
        m_num_command_queues == 2U,
        "AutoContext: the CCL sub-device needs the device opened with two command queues (collectives on the "
        "CCL sub-device issued from the compute queue stall the compute launches queued behind them)");
    TT_FATAL(
        (num_columns > 0U) != (num_rows > 0U),
        "AutoContext: the CCL sub-device is either the rightmost columns or the bottom rows, got {} columns and {} "
        "rows",
        num_columns,
        num_rows);
    auto& device = get_device();
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t grid_x = static_cast<uint32_t>(grid.x);
    const uint32_t grid_y = static_cast<uint32_t>(grid.y);
    TT_FATAL(num_columns < grid_x && num_rows < grid_y, "AutoContext: CCL sub-device must leave compute cores");
    // The CCL kernels take (workers + 1 mux) x 2 directions cores per link and pick 1, 2 or 4 workers
    // by what fits: 8 cores for 1 worker, 12 for 2, 20 for 4 (full speed). On a 12x10 Blackhole grid a
    // 10-core column gets 1 worker (measured 2.3x slower all-gather), a 12-core row 2, two columns 4.
    const uint32_t compute_x = grid_x - num_columns;
    const uint32_t compute_y = grid_y - num_rows;

    const std::array<tt::tt_metal::CoreRangeSet, 1> compute_cores{tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{compute_x - 1U, compute_y - 1U}))};
    const tt::tt_metal::CoreRange ccl_range =
        num_columns > 0U
            ? tt::tt_metal::CoreRange(
                  tt::tt_metal::CoreCoord{compute_x, 0}, tt::tt_metal::CoreCoord{grid_x - 1U, grid_y - 1U})
            : tt::tt_metal::CoreRange(
                  tt::tt_metal::CoreCoord{0, compute_y}, tt::tt_metal::CoreCoord{grid_x - 1U, grid_y - 1U});
    const std::array<tt::tt_metal::CoreRangeSet, 1> ccl_cores{tt::tt_metal::CoreRangeSet(ccl_range)};

    const std::array<tt::tt_metal::SubDevice, 2> sub_devices{
        tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(compute_cores)),
        tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(ccl_cores))};
    // local_l1_size 0: keep the single global allocator; the sub-devices only partition execution.
    const auto manager_id =
        device.create_sub_device_manager(ttsl::Span<const tt::tt_metal::SubDevice>(sub_devices), /* local_l1_size */ 0);
    device.load_sub_device_manager(manager_id);
    // Programs on a mesh command queue must lie inside one sub-device and nearly every op sizes its
    // core grid from compute_with_storage_grid_size(), so make the device report the compute
    // rectangle from now on. Leave the default stall group (all sub-devices) so host reads and
    // synchronizes wait for both.
    m_full_compute_grid = grid;
    device.set_compute_with_storage_grid_size_override(tt::tt_metal::CoreCoord{compute_x, compute_y});
    m_ccl_sub_device_id = tt::tt_metal::SubDeviceId{1};
}

bool AutoContext::has_ccl_sub_device() const {
    return m_ccl_sub_device_id.has_value();
}

std::optional<tt::tt_metal::SubDeviceId> AutoContext::ccl_sub_device_id() const {
    return m_ccl_sub_device_id;
}

tt::tt_metal::SubDeviceId AutoContext::compute_sub_device_id() const {
    return tt::tt_metal::SubDeviceId{0};
}

tt::tt_metal::CoreCoord AutoContext::full_compute_grid_size() {
    return m_full_compute_grid.value_or(get_device().compute_with_storage_grid_size());
}

void AutoContext::open_device(
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<int>& device_ids,
    size_t num_command_queues) {
    if (m_device) {
        throw std::runtime_error("open_device was called after the device was created.");
    }
    TT_FATAL(
        num_command_queues == 1 || num_command_queues == 2,
        "num_command_queues must be 1 or 2, got {}",
        num_command_queues);
    m_mesh_shape = mesh_shape;
    m_num_command_queues = num_command_queues;
    m_device = std::make_unique<core::MeshDevice>(m_mesh_shape, device_ids, num_command_queues);
}

size_t AutoContext::num_command_queues() const {
    return m_num_command_queues;
}

void AutoContext::close_profiler() {
    m_profiler = nullptr;
}

void AutoContext::close_device() {
    m_device = nullptr;
    // Drop the process-global fabric config that open_device_mesh may have
    // installed via enable_fabric(). Without this, fabric stays armed for
    // the remainder of the process and any subsequent default 1x1 open on a
    // host where mmio_chip_ids().size() != all_chip_ids().size() trips the
    // "Fabric is being used but Device i is not active" check in
    // tt_metal/impl/device/device_manager.cpp. Going TO DISABLED while no
    // devices are open is explicitly supported (see metal_env.cpp).
    ttnn_fixed::distributed::disable_fabric();
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
        if (config.enable_ddp && mesh_shape[axis] > 1U) {
            m_ddp_axis = axis++;
            m_num_ddp_devices = mesh_shape[m_ddp_axis.value()];
        }
        if (config.enable_cp && mesh_shape[axis] > 1U) {
            m_cp_axis = axis++;
            m_num_cp_devices = mesh_shape[m_cp_axis.value()];
        }
        if (config.enable_tp && mesh_shape[axis] > 1U) {
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
