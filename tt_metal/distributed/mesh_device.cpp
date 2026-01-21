// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <tt-logger/tt-logger.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_coord.hpp>
#include <mesh_device.hpp>
#include <mesh_device_view.hpp>
#include "distributed/mesh_device_impl.hpp"
#include <tt_stl/small_vector.hpp>
#include <sub_device.hpp>
#include "impl/sub_device/sub_device_impl.hpp"
#include <system_mesh.hpp>
#include <maybe_remote.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

#include "impl/allocator/allocator.hpp"
#include <tt_stl/assert.hpp>
#include "buffer.hpp"
#include "device/device_impl.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "host_api.hpp"
#include "mesh_config.hpp"
#include "mesh_trace.hpp"
#include "profiler_types.hpp"
#include <experimental/fabric/routing_table_generator.hpp>
#include "shape_base.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include "common/thread_pool.hpp"
#include "device/device_manager.hpp"
#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include "distributed/fd_mesh_command_queue.hpp"
#include "distributed/sd_mesh_command_queue.hpp"
#include "tracy/Tracy.hpp"
#include "tools/profiler/tt_metal_tracy.hpp"
#include <env_lib.hpp>

#include "allocator/l1_banking_allocator.hpp"
#include "debug/inspector/inspector.hpp"
#include "sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "context/metal_context.hpp"
#include "dispatch/system_memory_manager.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "mesh_device_view_impl.hpp"

namespace tt::tt_metal {
class CommandQueue;
class SystemMemoryManager;

namespace program_cache::detail {
struct ProgramCache;
}  // namespace program_cache::detail

}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
namespace {

int generate_unique_mesh_id() {
    static std::atomic<int> next_id{0};
    return next_id++;
}

std::shared_ptr<ThreadPool> create_default_thread_pool(const std::vector<IDevice*>& physical_devices) {
    // Bind the thread-pool to the physical devices being used.
    if (tt::parse_env("TT_MESH_PASS_THROUGH_THREAD_POOL", false) || physical_devices.size() == 1) {
        return create_passthrough_thread_pool();
    }
    return create_device_bound_thread_pool(physical_devices);
}

// Helper function to verify all devices in the MeshDevice have the same value
template <typename F>
decltype(auto) validate_and_get_reference_value(
    const std::vector<IDevice*>& devices, F&& func, const std::source_location& loc = std::source_location::current()) {
    if (devices.empty()) {
        TT_THROW("{} [{}:{}] failed: MeshDevice has no devices", loc.function_name(), loc.file_name(), loc.line());
    }

    // Forward the callable once to preserve its value category.
    auto&& callable = std::forward<F>(func);

    // Get reference to first device's value
    decltype(auto) reference_value = callable(devices.front());

    // Validate all other devices match
    for (auto it = devices.begin() + 1; it != devices.end(); ++it) {
        decltype(auto) current_value = callable(*it);
        if (current_value != reference_value) {
            TT_THROW(
                "{} [{}:{}] failed: Device at index {} returned value that differs from reference. "
                "Expected: {}, Actual: {}",
                loc.function_name(),
                loc.file_name(),
                loc.line(),
                std::distance(devices.begin(), it),
                reference_value,
                current_value);
        }
    }
    return reference_value;
}

// Returns offset of the mesh device view in the system mesh.
MeshCoordinate compute_system_mesh_offset(const MeshDeviceView& view) {
    const auto origin_fabric_node_id = view.get_fabric_node_id(MeshCoordinate::zero_coordinate(view.shape().dims()));
    const auto system_mesh_shape = SystemMesh::instance().shape();
    for (const auto& coord : MeshCoordinateRange(system_mesh_shape)) {
        if (coord.to_linear_index(system_mesh_shape) == origin_fabric_node_id.chip_id) {
            return coord;
        }
    }
    TT_THROW("Failed to find offset for mesh device view");
}

}  // namespace

MeshDeviceImpl::ScopedDevices::ScopedDevices(
    const std::vector<MaybeRemote<int>>& all_device_ids,
    const std::vector<MaybeRemote<int>>& active_device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    size_t worker_l1_size,
    const DispatchCoreConfig& dispatch_core_config) {
    auto local_devices = extract_locals(all_device_ids);
    opened_local_devices_ = tt_metal::detail::CreateDevices(
        local_devices,
        num_command_queues,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        /* init_profiler */ false,
        /* ignored */ true,
        /* initialize_fabric_and_dispatch_fw */ false);

    for (auto device_id : active_device_ids) {
        if (device_id.is_local()) {
            auto* device = opened_local_devices_.at(*device_id);
            devices_.push_back(MaybeRemoteDevice::local(device));
        } else {
            devices_.push_back(MaybeRemoteDevice::remote());
        }
    }
}

MeshDeviceImpl::ScopedDevices::~ScopedDevices() {
    if (!opened_local_devices_.empty()) {
        std::vector<IDevice*> devices_to_close;
        devices_to_close.reserve(opened_local_devices_.size());
        for (auto& [id, device] : opened_local_devices_) {
            devices_to_close.push_back(device);
        }
        tt_metal::MetalContext::instance().device_manager()->close_devices(devices_to_close, /*skip_synchronize=*/true);
    }
}

const std::map<ChipId, IDevice*>& MeshDeviceImpl::ScopedDevices::opened_local_devices() const {
    return opened_local_devices_;
}

const std::vector<MaybeRemote<IDevice*>>& MeshDeviceImpl::ScopedDevices::root_devices() const { return devices_; }

uint8_t MeshDeviceImpl::num_hw_cqs() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->num_hw_cqs(); });
}

bool MeshDeviceImpl::is_initialized() const {
    // TODO: Revisit whether we can simplify this when `MeshDevice` initialization isn't so coupled
    // with individual device initialization.
    if (!is_internal_state_initialized) {
        return false;
    }
    if (!scoped_devices_) {
        return false;
    }
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->is_initialized(); });
}

uint32_t MeshDeviceImpl::l1_size_per_core() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->l1_size_per_core(); });
}

uint32_t MeshDeviceImpl::dram_size_per_channel() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->dram_size_per_channel(); });
}

IDevice* MeshDeviceImpl::reference_device() const { return this->get_devices().at(0); }

// NOLINTNEXTLINE(readability-make-member-function-const)
void MeshDeviceImpl::mark_allocations_unsafe() { this->allocator_impl()->mark_allocations_unsafe(); }

// NOLINTNEXTLINE(readability-make-member-function-const)
void MeshDeviceImpl::mark_allocations_safe() { this->allocator_impl()->mark_allocations_safe(); }

MeshDeviceImpl::MeshDeviceImpl(
    std::shared_ptr<ScopedDevices> mesh_handle,
    std::unique_ptr<MeshDeviceView> mesh_device_view,
    std::shared_ptr<MeshDevice> parent_mesh) :
    scoped_devices_(std::move(mesh_handle)),
    mesh_id_(generate_unique_mesh_id()),
    view_(std::move(mesh_device_view)),
    parent_mesh_(std::move(parent_mesh)),
    dispatch_thread_pool_(create_default_thread_pool(extract_locals(scoped_devices_->root_devices()))),
    reader_thread_pool_(create_default_thread_pool(extract_locals(scoped_devices_->root_devices()))),
    program_cache_(std::make_unique<program_cache::detail::ProgramCache>()) {
    Inspector::mesh_device_created(this, parent_mesh_ ? std::make_optional(parent_mesh_->id()) : std::nullopt);
    const auto& mpi_context = MetalContext::instance().global_distributed_context();
    distributed_context_ =
        mpi_context.split(distributed::multihost::Color(id()), distributed::multihost::Key(*mpi_context.rank()));
}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    return MeshDeviceImpl::create(
        config,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size);
}

std::shared_ptr<MeshDevice> MeshDeviceImpl::create(
    const MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    const auto& mesh_graph = MetalContext::instance().get_control_plane().get_mesh_graph();
    auto [scoped_devices, fabric_node_ids, mesh_shape] =
        [&]() -> std::tuple<std::shared_ptr<ScopedDevices>, std::vector<tt::tt_fabric::FabricNodeId>, MeshShape> {
        if (config.physical_device_ids().empty()) {
            auto mapped_devices = SystemMesh::instance().get_mapped_devices(config.mesh_shape(), config.offset());
            // Validate that none of the fabric node IDs are on switch meshes
            for (const auto& fabric_node_id : mapped_devices.fabric_node_ids) {
                TT_FATAL(
                    !mesh_graph.is_switch_mesh(fabric_node_id.mesh_id),
                    "Cannot create devices on tt-switch meshes. Fabric node {} maps to mesh_id {} which is a switch. "
                    "Use get_compute_mesh_ids() to get valid compute mesh IDs.",
                    fabric_node_id,
                    *fabric_node_id.mesh_id);
            }
            auto mapped_devices_full_system_device_ids =
                (*MetalContext::instance().global_distributed_context().size() > 1)
                    ? SystemMesh::instance().get_mapped_devices(std::nullopt).device_ids
                    : mapped_devices.device_ids;
            return std::make_tuple(
                std::make_shared<MeshDeviceImpl::ScopedDevices>(
                    mapped_devices_full_system_device_ids,
                    mapped_devices.device_ids,
                    l1_small_size,
                    trace_region_size,
                    num_command_queues,
                    worker_l1_size,
                    dispatch_core_config),
                mapped_devices.fabric_node_ids,
                mapped_devices.mesh_shape);
        }  // Initialize fabric node ids manually.
        // TODO: #22087 - Remove this code path.
        std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
        TT_FATAL(config.mesh_shape().has_value(), "Mesh shape must be provided when physical device ids are supplied");
        const auto& supplied_ids = config.physical_device_ids();
        for (int supplied_id : supplied_ids) {
            auto fabric_node_id =
                MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(supplied_id);
            TT_FATAL(
                !mesh_graph.is_switch_mesh(fabric_node_id.mesh_id),
                "Cannot create devices on tt-switch meshes. Device {} maps to mesh_id {} which is a switch. "
                "Use get_compute_mesh_ids() to get valid compute mesh IDs.",
                supplied_id,
                *fabric_node_id.mesh_id);
            fabric_node_ids.push_back(fabric_node_id);
        }
        auto mapped_devices_full_system_device_ids =
            (*MetalContext::instance().global_distributed_context().size() > 1)
                ? SystemMesh::instance().get_mapped_devices(std::nullopt).device_ids
                : wrap_to_maybe_remote(supplied_ids);
        return std::make_tuple(
            std::make_shared<ScopedDevices>(
                mapped_devices_full_system_device_ids,
                wrap_to_maybe_remote(supplied_ids),
                l1_small_size,
                trace_region_size,
                num_command_queues,
                worker_l1_size,
                dispatch_core_config),
            fabric_node_ids,
            config.mesh_shape().value());
    }();

    // Make a copy because we std::move the scoped_devices when creating MeshDeviceImpl
    const auto root_devices = scoped_devices->root_devices();

    auto mesh_device = std::shared_ptr<MeshDevice>(new MeshDevice());
    mesh_device->pimpl_ = std::make_unique<MeshDeviceImpl>(
        std::move(scoped_devices),
        std::make_unique<MeshDeviceView>(mesh_shape, root_devices, fabric_node_ids),
        std::shared_ptr<MeshDevice>());

    mesh_device->initialize(num_command_queues, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap);

    // TODO #20966: Remove these calls
    for (auto* device : extract_locals(root_devices)) {
        dynamic_cast<Device*>(device)->set_mesh_device(mesh_device);
    }

    // Wait for all ranks to finish initializing the mesh device before proceeding.
    mesh_device->pimpl_->distributed_context_->barrier();

    // The Device Profiler must be initialized before Fabric is loaded on the Cluster
    tt_metal::MetalContext::instance().device_manager()->init_profiler();
    tt_metal::MetalContext::instance().device_manager()->initialize_fabric_and_dispatch_fw();
    return mesh_device;
}

void MeshDeviceImpl::enqueue_to_thread_pool(std::function<void()>&& f) { dispatch_thread_pool_->enqueue(std::move(f)); }

void MeshDeviceImpl::wait_for_thread_pool() { dispatch_thread_pool_->wait(); }

std::map<int, std::shared_ptr<MeshDevice>> MeshDevice::create_unit_meshes(
    const std::vector<int>& device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    return MeshDeviceImpl::create_unit_meshes(
        device_ids,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size);
}

std::map<int, std::shared_ptr<MeshDevice>> MeshDeviceImpl::create_unit_meshes(
    const std::vector<int>& device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> /*l1_bank_remap*/,
    size_t worker_l1_size) {
    // Validate all devices are on compute meshes (not switches) before creating any resources
    const auto& mesh_graph = MetalContext::instance().get_control_plane().get_mesh_graph();
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    fabric_node_ids.reserve(device_ids.size());
    for (const auto& device_id : device_ids) {
        auto fabric_node_id =
            MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(device_id);
        TT_FATAL(
            !mesh_graph.is_switch_mesh(fabric_node_id.mesh_id),
            "Cannot create devices on tt-switch meshes. Device {} maps to mesh_id {} which is a switch. "
            "Use get_compute_mesh_ids() to get valid compute mesh IDs.",
            device_id,
            *fabric_node_id.mesh_id);
        fabric_node_ids.push_back(fabric_node_id);
    }

    // Now create ScopedDevices after validation passes
    auto mapped_devices_full_system_device_ids =
        (*MetalContext::instance().global_distributed_context().size() > 1)
            ? SystemMesh::instance().get_mapped_devices(std::nullopt).device_ids
            : wrap_to_maybe_remote(device_ids);
    auto scoped_devices = std::make_shared<MeshDeviceImpl::ScopedDevices>(
        mapped_devices_full_system_device_ids,
        wrap_to_maybe_remote(device_ids),
        l1_small_size,
        trace_region_size,
        num_command_queues,
        worker_l1_size,
        dispatch_core_config);

    // Make a copy because we std::move the scoped_devices when creating MeshDeviceImpl
    const auto root_devices = scoped_devices->root_devices();

    auto mesh_device = std::shared_ptr<MeshDevice>(new MeshDevice());
    mesh_device->pimpl_ = std::make_unique<MeshDeviceImpl>(
        std::move(scoped_devices),
        std::make_unique<MeshDeviceView>(MeshShape(1, device_ids.size()), root_devices, fabric_node_ids),
        std::shared_ptr<MeshDevice>());

    auto submeshes = mesh_device->create_submeshes(MeshShape(1, 1));
    TT_FATAL(
        device_ids.size() == submeshes.size(),
        "Created an unexpected number of submeshes: {} instead of {}",
        submeshes.size(),
        device_ids.size());
    std::map<int, std::shared_ptr<MeshDevice>> result;
    for (size_t i = 0; i < device_ids.size(); i++) {
        result[device_ids[i]] = submeshes[i];
    }

    // Wait for all ranks to finish initializing the mesh device before proceeding.
    mesh_device->pimpl_->distributed_context_->barrier();

    // The Device Profiler must be initialized before Fabric is loaded on the Cluster
    tt_metal::MetalContext::instance().device_manager()->init_profiler();
    tt_metal::MetalContext::instance().device_manager()->initialize_fabric_and_dispatch_fw();
    return result;
}

std::shared_ptr<MeshDevice> MeshDevice::create_unit_mesh(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    return MeshDeviceImpl::create_unit_mesh(
        device_id,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size);
}

std::shared_ptr<MeshDevice> MeshDeviceImpl::create_unit_mesh(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    return create_unit_meshes(
               {device_id},
               l1_small_size,
               trace_region_size,
               num_command_queues,
               dispatch_core_config,
               l1_bank_remap,
               worker_l1_size)
        .at(device_id);
}

std::shared_ptr<MeshDevice> MeshDeviceImpl::create_submesh(
    const std::shared_ptr<MeshDevice>& parent_mesh,
    const MeshShape& submesh_shape,
    const std::optional<MeshCoordinate>& offset) {
    auto lock_api = this->lock_api();
    TT_FATAL(
        std::all_of(submesh_shape.cbegin(), submesh_shape.cend(), [](size_t dim) { return dim > 0; }),
        "Invalid submesh shape: ({}). All dimensions must be positive.",
        submesh_shape);
    TT_FATAL(
        submesh_shape.dims() == view_->shape().dims(),
        "Submesh shape {} and mesh device shape {} must have the same number of dimensions.",
        submesh_shape,
        view_->shape());

    const MeshCoordinate offset_coord = [&offset, &submesh_shape]() {
        if (offset.has_value()) {
            TT_FATAL(
                submesh_shape.dims() == offset->dims(),
                "Submesh shape {} and offset {} must have the same number of dimensions.",
                submesh_shape,
                *offset);
            return *offset;
        }
        return MeshCoordinate::zero_coordinate(submesh_shape.dims());
    }();

    tt::stl::SmallVector<uint32_t> end_coords;
    for (size_t i = 0; i < submesh_shape.dims(); i++) {
        TT_FATAL(
            offset_coord[i] + submesh_shape[i] - 1 < view_->shape()[i],
            "Submesh shape {} and offset {} does not fit within parent mesh ({}).",
            submesh_shape,
            offset,
            view_->shape());
        end_coords.push_back(offset_coord[i] + submesh_shape[i] - 1);
    }
    auto end_coordinate = MeshCoordinate(end_coords);

    // Create mesh device view for the submesh.
    std::vector<MaybeRemote<IDevice*>> submesh_devices;
    std::vector<tt::tt_fabric::FabricNodeId> submesh_fabric_node_ids;
    const MeshCoordinateRange submesh_range(offset_coord, end_coordinate);
    for (const auto& coord : submesh_range) {
        if (view_->impl().is_local(coord)) {
            submesh_devices.push_back(MaybeRemote<IDevice*>::local(view_->impl().get_device(coord)));
        } else {
            submesh_devices.push_back(MaybeRemote<IDevice*>::remote());
        }
        submesh_fabric_node_ids.push_back(view_->get_fabric_node_id(coord));
    }

    auto submesh = std::shared_ptr<MeshDevice>(new MeshDevice());
    submesh->pimpl_ = std::make_unique<MeshDeviceImpl>(
        scoped_devices_,
        std::make_unique<MeshDeviceView>(submesh_shape, submesh_devices, submesh_fabric_node_ids),
        parent_mesh);

    const auto& allocator_config = reference_device()->allocator_impl()->get_config();
    submesh->initialize(
        num_hw_cqs(),
        allocator_config.l1_small_size,
        allocator_config.trace_region_size,
        allocator_config.worker_l1_size,
        allocator_config.l1_bank_remap);

    // TODO #20966: Remove these calls
    if (!submesh->pimpl_->get_view().get_devices().empty()) {
        for (auto* device : submesh->pimpl_->get_devices()) {
            dynamic_cast<Device*>(device)->set_mesh_device(submesh);
        }
    }

    submeshes_.push_back(submesh);
    log_trace(LogMetal, "Instantiating submesh {}: {} with offset: {}", submesh->pimpl_->id(), submesh_shape, offset);
    if (!submesh->pimpl_->get_view().get_devices().empty()) {
        log_trace(
            LogMetal,
            "Submesh {} instantiated with {} devices",
            submesh->pimpl_->id(),
            submesh->pimpl_->get_devices().size());
    } else {
        log_trace(LogMetal, "Submesh {} instantiated with only remote devices", submesh->pimpl_->id());
    }
    return submesh;
}

std::vector<std::shared_ptr<MeshDevice>> MeshDeviceImpl::create_submeshes(
    const std::shared_ptr<MeshDevice>& parent_mesh, const MeshShape& submesh_shape) {
    // Calculate how many submeshes fit in each dimension.
    tt::stl::SmallVector<uint32_t> steps;
    for (size_t dim = 0; dim < shape().dims(); dim++) {
        TT_FATAL(
            shape()[dim] % submesh_shape[dim] == 0,
            "Shape {} is not divisible by submesh shape {} along dimension {}",
            shape(),
            submesh_shape,
            dim);
        uint32_t num_steps = shape()[dim] / submesh_shape[dim];
        steps.push_back(num_steps);
    }

    // Stamp `submesh_shape` along each dimension, `steps` number of times.
    std::vector<std::shared_ptr<MeshDevice>> submeshes;
    for (const auto& step_position : MeshCoordinateRange(MeshShape(steps))) {
        tt::stl::SmallVector<uint32_t> offset_coords;
        for (size_t dim = 0; dim < submesh_shape.dims(); dim++) {
            offset_coords.push_back(step_position[dim] * submesh_shape[dim]);
        }
        submeshes.push_back(create_submesh(parent_mesh, submesh_shape, MeshCoordinate(offset_coords)));
    }

    return submeshes;
}

MeshDeviceImpl::~MeshDeviceImpl() = default;

IDevice* MeshDeviceImpl::get_device(ChipId physical_device_id) const {
    for (auto* device : this->get_devices()) {
        if (device->id() == physical_device_id) {
            return device;
        }
    }
    TT_THROW("Physical Device ID: {} not found in assigned devices", physical_device_id);
}

std::vector<IDevice*> MeshDeviceImpl::get_devices() const {
    auto devices = view_->get_devices();
    TT_ASSERT(!devices.empty(), "Mesh Device should have at least 1 IDevice");
    return devices;
}

// TODO: Remove this function once we have a proper view interface
IDevice* MeshDeviceImpl::get_device(size_t row_idx, size_t col_idx) const {
    return get_device(MeshCoordinate{static_cast<uint32_t>(row_idx), static_cast<uint32_t>(col_idx)});
}

IDevice* MeshDeviceImpl::get_device(const MeshCoordinate& coord) const { return view_->impl().get_device(coord); }

tt_fabric::FabricNodeId MeshDeviceImpl::get_fabric_node_id(const MeshCoordinate& coord) const {
    return view_->get_fabric_node_id(coord);
}

MeshCommandQueue& MeshDeviceImpl::mesh_command_queue(std::optional<uint8_t> cq_id) const {
    auto id = cq_id.value_or(GetCurrentCommandQueueIdForThread());

    TT_FATAL(id < mesh_command_queues_.size(), "cq_id {} is out of range", id);
    const auto& command_queue = mesh_command_queues_[id];
    TT_FATAL(id == command_queue->id(), "MeshCommandQueue id mismatch, expected {}, got {}", id, command_queue->id());
    return *command_queue;
}

DeviceIds MeshDeviceImpl::get_device_ids() const {
    DeviceIds device_ids;
    for (auto* device : this->get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

size_t MeshDeviceImpl::num_devices() const { return view_->num_devices(); }

CoreCoord MeshDeviceImpl::compute_with_storage_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->compute_with_storage_grid_size(); });
}

tt::ARCH MeshDeviceImpl::arch() const { return tt_metal::MetalContext::instance().get_cluster().arch(); }

size_t MeshDeviceImpl::num_rows() const { return view_->num_rows(); }

size_t MeshDeviceImpl::num_cols() const { return view_->num_cols(); }

const MeshShape& MeshDeviceImpl::shape() const { return view_->shape(); }

bool MeshDeviceImpl::is_local(const MeshCoordinate& coord) const { return view_->impl().is_local(coord); }

void MeshDeviceImpl::reshape(const MeshShape& new_shape) {
    const auto num_devices = view_->shape().mesh_size();
    TT_FATAL(new_shape.mesh_size() == num_devices, "New shape must have the same number of devices as current shape");

    // MeshDeviceView requires devices to be provided as a 1D array in row-major order for the target mesh shape.
    // The physical connectivity between devices must be preserved when reshaping.
    //
    // Example:
    // Given 4 devices physically connected in a 2x2 grid like this:
    //   [0]--[1]
    //    |    |
    //   [3]--[2]
    //
    // For a 1x4 mesh shape:
    // - Devices must form a line: 0->1->2->3
    // - Row-major order will be: [0,1,2,3]
    //
    // For a 2x2 mesh shape:
    // - Preserves original 2x2 physical connectivity
    // - Row-major order will be: [0,1,3,2]
    std::unordered_set<tt::tt_fabric::FabricNodeId> current_fabric_nodes;
    for (const auto& coord : MeshCoordinateRange(view_->shape())) {
        current_fabric_nodes.insert(view_->get_fabric_node_id(coord));
    }

    // From an MxN mesh, we can always reduce rank to a 1xM*N Line mesh.
    // However, going from a Line mesh to an MxN mesh is not always possible.
    std::vector<MaybeRemote<IDevice*>> new_device_order;
    std::vector<tt::tt_fabric::FabricNodeId> new_fabric_node_ids;
    new_device_order.reserve(num_devices);
    new_fabric_node_ids.reserve(num_devices);
    if (new_shape.is_line_topology()) {
        auto line_coords = view_->get_line_coordinates();
        for (const auto& coord : line_coords) {
            new_device_order.push_back(
                view_->impl().is_local(coord) ? MaybeRemote<IDevice*>::local(this->get_device(coord))
                                              : MaybeRemote<IDevice*>::remote());
            new_fabric_node_ids.push_back(view_->get_fabric_node_id(coord));
        }
    } else {
        // Do our best at requesting a new set of mapped devices from system mesh, starting at the offset of the first
        // device in the original mesh.
        auto new_mapped_devices =
            SystemMesh::instance().get_mapped_devices(new_shape, compute_system_mesh_offset(*view_));
        for (int i = 0; i < new_mapped_devices.device_ids.size(); i++) {
            TT_FATAL(
                current_fabric_nodes.contains(new_mapped_devices.fabric_node_ids[i]),
                "User has requested a reshape of the MeshDevice to shape: {}, but it is not possible to form a "
                "physically connected mesh grid with the opened devices from the original shape: {}.",
                new_shape,
                view_->shape());
            new_device_order.push_back(
                new_mapped_devices.device_ids[i].is_local()
                    ? MaybeRemote<IDevice*>::local(get_device(*new_mapped_devices.device_ids[i]))
                    : MaybeRemote<IDevice*>::remote());
        }
        new_fabric_node_ids = std::move(new_mapped_devices.fabric_node_ids);
    }
    auto new_view = std::make_unique<MeshDeviceView>(new_shape, new_device_order, new_fabric_node_ids);
    view_ = std::move(new_view);
}

bool MeshDeviceImpl::close() {
    TT_THROW("close() is not supported on MeshDeviceImpl - use close_impl() instead");
    return false;
}

bool MeshDeviceImpl::close_impl(MeshDevice* pimpl_wrapper) {
    ZoneScoped;

    log_trace(tt::LogMetal, "Closing mesh device {}", this->id());

    if (this->is_initialized()) {
        ReadMeshDeviceProfilerResults(*pimpl_wrapper, ProfilerReadState::LAST_FD_READ);
    }

    if (distributed_context_) {
        // Wait for all ranks to be ready to close the mesh device before proceeding.
        distributed_context_->barrier();
    }

    // TODO #20966: Remove these calls
    for (auto* device : view_->get_devices()) {
        dynamic_cast<Device*>(device)->set_mesh_device(parent_mesh_);
    }

    // Only one mesh device can use a CQ on a physical device at a time, or else teardown or some other operation will
    // hang. Validate this.
    for (uint32_t cq_id = 0; cq_id < mesh_command_queues_.size(); cq_id++) {
        if (mesh_command_queues_[cq_id]->in_use()) {
            auto parent_mesh = get_parent_mesh();
            if (parent_mesh) {
                auto parent_mesh_id = parent_mesh->impl().get_parent_mesh_id_with_in_use_cq(cq_id);
                if (parent_mesh_id) {
                    TT_THROW(
                        "MeshDevice cq ID {} is in use by parent mesh ID {} during close of mesh ID {}",
                        cq_id,
                        *parent_mesh_id,
                        id());
                }
            }

            for (const auto& submesh : submeshes_) {
                if (auto submesh_ptr = submesh.lock()) {
                    auto child_mesh_id = submesh_ptr->impl().get_child_mesh_id_with_in_use_cq(cq_id);
                    if (child_mesh_id) {
                        TT_THROW(
                            "MeshDevice cq ID {} is in use by child submesh ID {} during close of mesh ID {}",
                            cq_id,
                            *child_mesh_id,
                            id());
                    }
                }
            }
        }
    }

    mesh_command_queues_.clear();
    sub_device_manager_tracker_.reset();
    scoped_devices_.reset();
    parent_mesh_.reset();
    is_internal_state_initialized = false;
    if (distributed_context_) {
        distributed_context_.reset();
    }

    return true;
}

std::optional<int> MeshDeviceImpl::get_parent_mesh_id_with_in_use_cq(uint32_t cq_id) const {
    if (cq_id < mesh_command_queues_.size() && mesh_command_queues_[cq_id]->in_use()) {
        return id();
    }
    if (parent_mesh_) {
        return parent_mesh_->impl().get_parent_mesh_id_with_in_use_cq(cq_id);
    }
    return std::nullopt;
}

std::optional<int> MeshDeviceImpl::get_child_mesh_id_with_in_use_cq(uint32_t cq_id) const {
    if (cq_id < mesh_command_queues_.size() && mesh_command_queues_[cq_id]->in_use()) {
        return id();
    }
    for (const auto& submesh : submeshes_) {
        if (auto submesh_ptr = submesh.lock()) {
            auto child_mesh_id = submesh_ptr->impl().get_child_mesh_id_with_in_use_cq(cq_id);
            if (child_mesh_id) {
                return child_mesh_id;
            }
        }
    }
    return std::nullopt;
}

std::string MeshDeviceImpl::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", this->num_rows(), this->num_cols(), this->num_devices());
}

const MeshDeviceView& MeshDeviceImpl::get_view() const {
    TT_FATAL(view_, "MeshDeviceView is not initialized");
    return *view_;
}

int MeshDeviceImpl::id() const { return mesh_id_; }
// For a mesh, build id is the same as the device id for the reference device
ChipId MeshDeviceImpl::build_id() const { return reference_device()->id(); }

bool MeshDeviceImpl::is_parent_mesh() const { return parent_mesh_ == nullptr; }

const std::shared_ptr<MeshDevice>& MeshDeviceImpl::get_parent_mesh() const { return parent_mesh_; }
std::vector<std::shared_ptr<MeshDevice>> MeshDeviceImpl::get_submeshes() const {
    std::vector<std::shared_ptr<MeshDevice>> result;
    result.reserve(submeshes_.size());
    for (const auto& weak_submesh : submeshes_) {
        if (auto submesh = weak_submesh.lock()) {
            result.push_back(std::move(submesh));
        }
    }
    return result;
}

void MeshDeviceImpl::enable_program_cache() {
    log_info(tt::LogMetal, "Enabling program cache on MeshDevice {}", this->id());
    program_cache_->enable();
}

void MeshDeviceImpl::clear_program_cache() {
    log_info(tt::LogMetal, "Clearing program cache on MeshDevice {}", this->id());
    program_cache_->clear();
}

void MeshDeviceImpl::disable_and_clear_program_cache() {
    log_info(tt::LogMetal, "Disabling and clearing program cache on MeshDevice {}", this->id());
    if (program_cache_->is_enabled()) {
        program_cache_->disable();
    }
    program_cache_->clear();
}

size_t MeshDeviceImpl::num_program_cache_entries() { return program_cache_->num_entries(); }

SubDeviceManagerId MeshDeviceImpl::create_sub_device_manager(
    std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto lock = lock_api();
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}

SubDeviceManagerId MeshDeviceImpl::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    auto lock = lock_api();
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}
void MeshDeviceImpl::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    auto lock = lock_api();
    sub_device_manager_tracker_->remove_sub_device_manager(sub_device_manager_id);
}
void MeshDeviceImpl::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    auto lock = lock_api();
    sub_device_manager_tracker_->load_sub_device_manager(sub_device_manager_id);
}
void MeshDeviceImpl::clear_loaded_sub_device_manager() {
    sub_device_manager_tracker_->clear_loaded_sub_device_manager();
}

CoreCoord MeshDeviceImpl::dram_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->dram_grid_size(); });
}

// Device property methods that can be delegated to reference device
CoreCoord MeshDeviceImpl::grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->grid_size(); });
}
CoreCoord MeshDeviceImpl::logical_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->logical_grid_size(); });
}
CoreCoord MeshDeviceImpl::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    TT_FATAL(num_devices() == 1, "virtual_noc0_coordinate() is only supported on unit MeshDevice.");
    return get_devices().front()->virtual_noc0_coordinate(noc_index, coord);
}
std::vector<CoreCoord> MeshDeviceImpl::worker_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_cores](const auto* device) {
        return device->worker_cores_from_logical_cores(logical_cores);
    });
}
std::vector<CoreCoord> MeshDeviceImpl::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    return get_devices().front()->get_optimal_dram_bank_to_logical_worker_assignment(noc);
}
CoreCoord MeshDeviceImpl::virtual_core_from_logical_core(
    const CoreCoord& logical_coord, const CoreType& core_type) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_coord, core_type](const auto* device) {
        return device->virtual_core_from_logical_core(logical_coord, core_type);
    });
}
CoreCoord MeshDeviceImpl::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->worker_core_from_logical_core(logical_core);
    });
}
CoreCoord MeshDeviceImpl::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return validate_and_get_reference_value(this->get_devices(), [ethernet_core](const auto* device) {
        return device->logical_core_from_ethernet_core(ethernet_core);
    });
}

// These methods require some change / or assert out for now
std::vector<CoreCoord> MeshDeviceImpl::ethernet_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_cores](const auto* device) {
        return device->ethernet_cores_from_logical_cores(logical_cores);
    });
}
CoreCoord MeshDeviceImpl::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->ethernet_core_from_logical_core(logical_core);
    });
}
std::unordered_set<CoreCoord> MeshDeviceImpl::get_active_ethernet_cores(bool /*skip_reserved_tunnel_cores*/) const {
    TT_THROW("get_active_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

std::unordered_set<CoreCoord> MeshDeviceImpl::get_inactive_ethernet_cores() const {
    TT_THROW("get_inactive_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDeviceImpl::is_inactive_ethernet_core(CoreCoord /*logical_core*/) const {
    TT_THROW("is_inactive_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::tuple<ChipId, CoreCoord> MeshDeviceImpl::get_connected_ethernet_core(CoreCoord /*eth_core*/) const {
    TT_THROW("get_connected_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDeviceImpl::is_active_ethernet_core(CoreCoord /*logical_core*/, bool /*skip_reserved_tunnel_cores*/) const {
    TT_THROW("is_active_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::vector<CoreCoord> MeshDeviceImpl::get_ethernet_sockets(ChipId /*connected_chip_id*/) const {
    TT_THROW("get_ethernet_sockets() is not supported on MeshDevice - use individual devices instead");
}

uint32_t MeshDeviceImpl::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    // Issue #19729: Return the maximum number of active ethernet cores across physical devices in the Mesh.
    TT_FATAL(*sub_device_id == 0, "Cannot query virtual ethernet cores per sub-device when using MeshDevice");
    return num_virtual_eth_cores_;
}

// Core and worker management methods (These are OK)
CoreRangeSet MeshDeviceImpl::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}

uint32_t MeshDeviceImpl::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()
        ->sub_device(sub_device_id)
        .impl()
        ->num_cores(core_type);
}

// Bank and memory management methods
int MeshDeviceImpl::num_dram_channels() const { return reference_device()->num_dram_channels(); }

CoreCoord MeshDeviceImpl::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return validate_and_get_reference_value(this->get_devices(), [dram_channel](const auto* device) {
        return device->logical_core_from_dram_channel(dram_channel);
    });
}
uint32_t MeshDeviceImpl::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->dram_channel_from_logical_core(logical_core);
    });
}
uint32_t MeshDeviceImpl::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    return validate_and_get_reference_value(this->get_devices(), [virtual_core](const auto* device) {
        return device->dram_channel_from_virtual_core(virtual_core);
    });
}

// Core management and network operations
const std::set<CoreCoord>& MeshDeviceImpl::ethernet_cores() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) -> const std::set<CoreCoord>& { return device->ethernet_cores(); });
}
const std::set<CoreCoord>& MeshDeviceImpl::storage_only_cores() const {
    return validate_and_get_reference_value(this->get_devices(), [](const auto* device) -> const std::set<CoreCoord>& {
        return device->storage_only_cores();
    });
}
uint32_t MeshDeviceImpl::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    return validate_and_get_reference_value(this->get_devices(), [noc_index, core](const auto* device) {
        return device->get_noc_unicast_encoding(noc_index, core);
    });
}
uint32_t MeshDeviceImpl::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    return validate_and_get_reference_value(this->get_devices(), [noc_index, cores](const auto* device) {
        return device->get_noc_multicast_encoding(noc_index, cores);
    });
}

// System memory and command queue management
SystemMemoryManager& MeshDeviceImpl::sysmem_manager() {
    TT_THROW("sysmem_manager() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->sysmem_manager();
}

CommandQueue& MeshDeviceImpl::command_queue(std::optional<uint8_t> cq_id) {
    TT_THROW("command_queue() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->command_queue(cq_id);
}

void MeshDeviceImpl::release_mesh_trace(const MeshTraceId& trace_id) {
    TracyTTMetalReleaseMeshTrace(this->get_device_ids(), *trace_id);

    sub_device_manager_tracker_->get_active_sub_device_manager()->release_trace(trace_id);

    // Only enable allocations once all captured traces are released
    if (this->trace_buffers_size_ == 0) {
        this->mark_allocations_safe();
    }
}

std::shared_ptr<MeshTraceBuffer> MeshDeviceImpl::get_mesh_trace(const MeshTraceId& trace_id) {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_trace(trace_id);
}

MeshTraceId MeshDeviceImpl::begin_mesh_trace(uint8_t cq_id) {
    auto trace_id = MeshTrace::next_id();
    this->begin_mesh_trace(cq_id, trace_id);
    return trace_id;
}

void MeshDeviceImpl::begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalBeginMeshTrace(this->get_device_ids(), *trace_id);
    TT_FATAL(
        !this->mesh_command_queues_[cq_id]->trace_id().has_value(),
        "CQ {} is already being used for tracing tid {}",
        (uint32_t)cq_id,
        *trace_id);
    this->mark_allocations_safe();
    // Create an empty trace buffer here. This will get initialized in end_trace
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(trace_id) == nullptr,
        "Trace already exists for tid {} on device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        active_sub_device_manager->id());
    auto& trace_buffer = active_sub_device_manager->create_trace(trace_id);
    this->mesh_command_queues_[cq_id]->record_begin(trace_id, trace_buffer->desc);
}

void MeshDeviceImpl::end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalEndMeshTrace(this->get_device_ids(), *trace_id);
    TT_FATAL(
        this->mesh_command_queues_[cq_id]->trace_id() == trace_id,
        "CQ {} is not being used for tracing tid {}",
        (uint32_t)cq_id,
        trace_id);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    auto trace_buffer = active_sub_device_manager->get_trace(trace_id);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        active_sub_device_manager->id());
    this->mesh_command_queues_[cq_id]->record_end();

    MeshTrace::populate_mesh_buffer(*(mesh_command_queues_[cq_id]), trace_buffer);
    this->mark_allocations_unsafe();
}

void MeshDeviceImpl::replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    ZoneScoped;
    TracyTTMetalReplayMeshTrace(this->get_device_ids(), *trace_id);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    const auto& trace_buffer = active_sub_device_manager->get_trace(trace_id);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on Mesh device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        *(active_sub_device_manager->id()));
    mesh_command_queues_[cq_id]->enqueue_trace(trace_id, blocking);
}

uint32_t MeshDeviceImpl::get_trace_buffers_size() const { return trace_buffers_size_; }
void MeshDeviceImpl::set_trace_buffers_size(uint32_t size) { trace_buffers_size_ = size; }

bool MeshDeviceImpl::initialize(
    const uint8_t /*num_hw_cqs*/,
    size_t /*l1_small_size*/,
    size_t /*trace_region_size*/,
    size_t /*worker_l1_size*/,
    tt::stl::Span<const std::uint32_t> /*l1_bank_remap*/,
    bool /*minimal*/) {
    TT_THROW("initialize() is not supported on MeshDeviceImpl - use initialize_impl() instead");
    return false;
}

// Dispatch and initialization
bool MeshDeviceImpl::initialize_impl(
    MeshDevice* pimpl_wrapper,
    const uint8_t /*num_hw_cqs*/,
    size_t /*l1_small_size*/,
    size_t /*trace_region_size*/,
    size_t /*worker_l1_size*/,
    tt::stl::Span<const std::uint32_t> /*l1_bank_remap*/,
    bool /*minimal*/) {
    TT_FATAL(!this->is_initialized(), "MeshDevice is already initialized!");

    // If the mesh device has no local devices, do not attempt to initialize it.
    if (view_->get_devices().empty()) {
        return false;
    }

    // For MeshDevice, we support uniform sub-devices across all devices and we do not support ethernet subdevices.
    const auto& compute_grid_size = this->compute_with_storage_grid_size();
    auto sub_devices = {
        SubDevice(std::array{CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}))})};

    // Resource shared across mesh command queues.
    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);

    const auto& allocator = reference_device()->allocator_impl();
    // SubDeviceManagerTracker needs a MeshDevice pointer.
    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        pimpl_wrapper, std::make_unique<L1BankingAllocator>(allocator->get_config()), sub_devices);
    // Issue #19729: Store the maximum number of active ethernet cores across opened physical devices in the Mesh
    // as the number of virtual ethernet cores seen by the MeshDevice
    num_virtual_eth_cores_ =
        tt_metal::MetalContext::instance().device_manager()->get_max_num_eth_cores_across_all_devices();
    mesh_command_queues_.reserve(this->num_hw_cqs());
    if (MetalContext::instance().rtoptions().get_fast_dispatch()) {
        for (std::size_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
            mesh_command_queues_.push_back(std::make_unique<FDMeshCommandQueue>(
                pimpl_wrapper,
                cq_id,
                dispatch_thread_pool_,
                reader_thread_pool_,
                cq_shared_state,
                std::bind(&MeshDeviceImpl::lock_api, this)));
        }
    } else {
        for (std::size_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
            mesh_command_queues_.push_back(
                std::make_unique<SDMeshCommandQueue>(pimpl_wrapper, cq_id, std::bind(&MeshDeviceImpl::lock_api, this)));
        }
    }
    Inspector::mesh_device_initialized(this);
    is_internal_state_initialized = true;
    return true;
}

void MeshDeviceImpl::init_command_queue_host() {
    TT_THROW("init_command_queue_host() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_host();
}
void MeshDeviceImpl::init_command_queue_device() {
    TT_THROW("init_command_queue_device() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_device();
}
bool MeshDeviceImpl::compile_fabric() {
    TT_THROW("compile_fabric() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->compile_fabric();
}
void MeshDeviceImpl::configure_fabric() {
    TT_THROW("configure_fabric() is not supported on MeshDevice - use individual devices instead");
    reference_device()->configure_fabric();
}
void MeshDeviceImpl::init_fabric() {
    TT_THROW("init_fabric_program() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_fabric();
}

program_cache::detail::ProgramCache& MeshDeviceImpl::get_program_cache() { return *program_cache_; }
HalProgrammableCoreType MeshDeviceImpl::get_programmable_core_type(CoreCoord virtual_core) const {
    return reference_device()->get_programmable_core_type(virtual_core);
}

HalMemType MeshDeviceImpl::get_mem_type_of_core(CoreCoord virtual_core) const {
    return reference_device()->get_mem_type_of_core(virtual_core);
}

// Methods for SubDevice Management
bool MeshDeviceImpl::has_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->has_noc_mcast_txns(sub_device_id);
}
uint8_t MeshDeviceImpl::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_unicast_txns(sub_device_id);
}
uint8_t MeshDeviceImpl::noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data) const {
    if (unicast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_unicast_data_start_index(
            sub_device_id);
    }
    return 0;
}
SubDeviceManagerId MeshDeviceImpl::get_active_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->id();
}
SubDeviceManagerId MeshDeviceImpl::get_default_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->id();
}
CoreCoord MeshDeviceImpl::virtual_program_dispatch_core(uint8_t cq_id) const {
    return validate_and_get_reference_value(
        this->get_devices(), [cq_id](const auto* device) { return device->virtual_program_dispatch_core(cq_id); });
}
const std::vector<SubDeviceId>& MeshDeviceImpl::get_sub_device_ids() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_ids();
}
const std::vector<SubDeviceId>& MeshDeviceImpl::get_sub_device_stall_group() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_stall_group();
}
void MeshDeviceImpl::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_manager_tracker_->get_active_sub_device_manager()->set_sub_device_stall_group(sub_device_ids);
}
void MeshDeviceImpl::reset_sub_device_stall_group() {
    sub_device_manager_tracker_->get_active_sub_device_manager()->reset_sub_device_stall_group();
}

uint32_t MeshDeviceImpl::num_sub_devices() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices();
}

bool MeshDeviceImpl::is_mmio_capable() const {
    TT_THROW("is_mmio_capable() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->is_mmio_capable();
}

void MeshDeviceImpl::quiesce_internal() {
    TT_FATAL(
        get_active_sub_device_manager_id() == get_default_sub_device_manager_id(),
        "Cannot quiesce when non-default sub-device manager is active");
    for (const auto& submesh : submeshes_) {
        if (auto submesh_ptr = submesh.lock()) {
            submesh_ptr->quiesce_devices();
        }
    }
    bool have_reset_launch_msg_state = false;
    for (auto& command_queue : mesh_command_queues_) {
        command_queue->wait_for_completion(!have_reset_launch_msg_state);
        have_reset_launch_msg_state = true;
    }
    for (auto& command_queue : mesh_command_queues_) {
        command_queue->finish_and_reset_in_use();
    }
}

void MeshDeviceImpl::quiesce_devices() {
    quiesce_internal();
    for (auto& command_queue : mesh_command_queues_) {
        for (auto& device : get_devices()) {
            TT_ASSERT(
                device->sysmem_manager().get_last_completed_event(command_queue->id()) == 0,
                "Last completed event is not 0");
            TT_ASSERT(device->sysmem_manager().get_current_event(command_queue->id()) == 0, "Current event is not 0");
        }
    }
}

// Allocator methods
std::optional<DeviceAddr> MeshDeviceImpl::lowest_occupied_compute_l1_address() const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address();
}

std::optional<DeviceAddr> MeshDeviceImpl::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address(sub_device_ids);
}

const std::unique_ptr<AllocatorImpl>& MeshDeviceImpl::allocator_impl() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->allocator(SubDeviceId{0});
}

const std::unique_ptr<Allocator>& MeshDeviceImpl::allocator() const { return this->allocator_impl()->view(); }

const std::unique_ptr<AllocatorImpl>& MeshDeviceImpl::allocator_impl(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->allocator(sub_device_id);
}

const std::unique_ptr<Allocator>& MeshDeviceImpl::allocator(SubDeviceId sub_device_id) const {
    return this->allocator_impl(sub_device_id)->view();
}

std::shared_ptr<distributed::MeshDevice> MeshDeviceImpl::get_mesh_device() {
    // This should be called from MeshDevice, not MeshDeviceImpl
    TT_THROW("get_mesh_device() should not be called on MeshDeviceImpl directly");
    return nullptr;
}

MeshDevice::~MeshDevice() {
    Inspector::mesh_device_destroyed(this->pimpl_.get());
    pimpl_->close_impl(this);
}

// MeshDevice PIMPL forwarding methods
tt::ARCH MeshDevice::arch() const { return pimpl_->arch(); }
int MeshDevice::id() const { return pimpl_->id(); }
ChipId MeshDevice::build_id() const { return pimpl_->build_id(); }
uint8_t MeshDevice::num_hw_cqs() const { return pimpl_->num_hw_cqs(); }
bool MeshDevice::is_initialized() const { return pimpl_->is_initialized(); }
int MeshDevice::num_dram_channels() const { return pimpl_->num_dram_channels(); }
uint32_t MeshDevice::l1_size_per_core() const { return pimpl_->l1_size_per_core(); }
uint32_t MeshDevice::dram_size_per_channel() const { return pimpl_->dram_size_per_channel(); }
CoreCoord MeshDevice::grid_size() const { return pimpl_->grid_size(); }
CoreCoord MeshDevice::logical_grid_size() const { return pimpl_->logical_grid_size(); }
CoreCoord MeshDevice::dram_grid_size() const { return pimpl_->dram_grid_size(); }
CoreCoord MeshDevice::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    return pimpl_->virtual_noc0_coordinate(noc_index, coord);
}
std::vector<CoreCoord> MeshDevice::worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    return pimpl_->worker_cores_from_logical_cores(logical_cores);
}
std::vector<CoreCoord> MeshDevice::ethernet_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return pimpl_->ethernet_cores_from_logical_cores(logical_cores);
}
std::vector<CoreCoord> MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    return pimpl_->get_optimal_dram_bank_to_logical_worker_assignment(noc);
}
CoreCoord MeshDevice::virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const {
    return pimpl_->virtual_core_from_logical_core(logical_coord, core_type);
}
CoreCoord MeshDevice::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return pimpl_->worker_core_from_logical_core(logical_core);
}
CoreCoord MeshDevice::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    return pimpl_->ethernet_core_from_logical_core(logical_core);
}
CoreCoord MeshDevice::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return pimpl_->logical_core_from_ethernet_core(ethernet_core);
}
std::unordered_set<CoreCoord> MeshDevice::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return pimpl_->get_active_ethernet_cores(skip_reserved_tunnel_cores);
}
std::unordered_set<CoreCoord> MeshDevice::get_inactive_ethernet_cores() const {
    return pimpl_->get_inactive_ethernet_cores();
}
bool MeshDevice::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    return pimpl_->is_active_ethernet_core(logical_core, skip_reserved_tunnel_cores);
}
std::tuple<ChipId, CoreCoord> MeshDevice::get_connected_ethernet_core(CoreCoord eth_core) const {
    return pimpl_->get_connected_ethernet_core(eth_core);
}
std::vector<CoreCoord> MeshDevice::get_ethernet_sockets(ChipId connected_chip_id) const {
    return pimpl_->get_ethernet_sockets(connected_chip_id);
}
bool MeshDevice::is_inactive_ethernet_core(CoreCoord logical_core) const {
    return pimpl_->is_inactive_ethernet_core(logical_core);
}
uint32_t MeshDevice::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    return pimpl_->num_virtual_eth_cores(sub_device_id);
}
CoreCoord MeshDevice::compute_with_storage_grid_size() const { return pimpl_->compute_with_storage_grid_size(); }
CoreRangeSet MeshDevice::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return pimpl_->worker_cores(core_type, sub_device_id);
}
uint32_t MeshDevice::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return pimpl_->num_worker_cores(core_type, sub_device_id);
}
const std::unique_ptr<Allocator>& MeshDevice::allocator() const { return pimpl_->allocator(); }
const std::unique_ptr<Allocator>& MeshDevice::allocator(SubDeviceId sub_device_id) const {
    return pimpl_->allocator(sub_device_id);
}
const std::unique_ptr<AllocatorImpl>& MeshDevice::allocator_impl() const { return pimpl_->allocator_impl(); }
const std::unique_ptr<AllocatorImpl>& MeshDevice::allocator_impl(SubDeviceId sub_device_id) const {
    return pimpl_->allocator_impl(sub_device_id);
}
CoreCoord MeshDevice::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return pimpl_->logical_core_from_dram_channel(dram_channel);
}
uint32_t MeshDevice::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return pimpl_->dram_channel_from_logical_core(logical_core);
}
uint32_t MeshDevice::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    return pimpl_->dram_channel_from_virtual_core(virtual_core);
}
std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address() const {
    return pimpl_->lowest_occupied_compute_l1_address();
}
std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return pimpl_->lowest_occupied_compute_l1_address(sub_device_ids);
}
const std::set<CoreCoord>& MeshDevice::ethernet_cores() const { return pimpl_->ethernet_cores(); }
const std::set<CoreCoord>& MeshDevice::storage_only_cores() const { return pimpl_->storage_only_cores(); }
uint32_t MeshDevice::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    return pimpl_->get_noc_unicast_encoding(noc_index, core);
}
uint32_t MeshDevice::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    return pimpl_->get_noc_multicast_encoding(noc_index, cores);
}
SystemMemoryManager& MeshDevice::sysmem_manager() { return pimpl_->sysmem_manager(); }
CommandQueue& MeshDevice::command_queue(std::optional<uint8_t> cq_id) { return pimpl_->command_queue(cq_id); }
MeshTraceId MeshDevice::begin_mesh_trace(uint8_t cq_id) { return pimpl_->begin_mesh_trace(cq_id); }
void MeshDevice::begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    pimpl_->begin_mesh_trace(cq_id, trace_id);
}
void MeshDevice::end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) { pimpl_->end_mesh_trace(cq_id, trace_id); }
void MeshDevice::replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    pimpl_->replay_mesh_trace(cq_id, trace_id, blocking);
}
void MeshDevice::release_mesh_trace(const MeshTraceId& trace_id) { pimpl_->release_mesh_trace(trace_id); }
std::shared_ptr<MeshTraceBuffer> MeshDevice::get_mesh_trace(const MeshTraceId& trace_id) {
    return pimpl_->get_mesh_trace(trace_id);
}
uint32_t MeshDevice::get_trace_buffers_size() const { return pimpl_->get_trace_buffers_size(); }
void MeshDevice::set_trace_buffers_size(uint32_t size) { pimpl_->set_trace_buffers_size(size); }
bool MeshDevice::initialize(
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal) {
    return pimpl_->initialize_impl(
        this, num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal);
}
void MeshDevice::init_command_queue_host() { pimpl_->init_command_queue_host(); }
void MeshDevice::init_command_queue_device() { pimpl_->init_command_queue_device(); }
bool MeshDevice::compile_fabric() { return pimpl_->compile_fabric(); }
void MeshDevice::configure_fabric() { pimpl_->configure_fabric(); }
void MeshDevice::init_fabric() { pimpl_->init_fabric(); }
bool MeshDevice::close() { return pimpl_->close_impl(this); }
void MeshDevice::enable_program_cache() { pimpl_->enable_program_cache(); }
void MeshDevice::clear_program_cache() { pimpl_->clear_program_cache(); }
void MeshDevice::disable_and_clear_program_cache() { pimpl_->disable_and_clear_program_cache(); }
program_cache::detail::ProgramCache& MeshDevice::get_program_cache() { return pimpl_->get_program_cache(); }
std::size_t MeshDevice::num_program_cache_entries() { return pimpl_->num_program_cache_entries(); }
HalProgrammableCoreType MeshDevice::get_programmable_core_type(CoreCoord virtual_core) const {
    return pimpl_->get_programmable_core_type(virtual_core);
}
HalMemType MeshDevice::get_mem_type_of_core(CoreCoord virtual_core) const {
    return pimpl_->get_mem_type_of_core(virtual_core);
}
bool MeshDevice::has_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return pimpl_->has_noc_mcast_txns(sub_device_id);
}
uint8_t MeshDevice::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return pimpl_->num_noc_unicast_txns(sub_device_id);
}
uint8_t MeshDevice::noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data) const {
    return pimpl_->noc_data_start_index(sub_device_id, unicast_data);
}
SubDeviceManagerId MeshDevice::get_active_sub_device_manager_id() const {
    return pimpl_->get_active_sub_device_manager_id();
}
SubDeviceManagerId MeshDevice::get_default_sub_device_manager_id() const {
    return pimpl_->get_default_sub_device_manager_id();
}
SubDeviceManagerId MeshDevice::create_sub_device_manager(
    std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return pimpl_->create_sub_device_manager(sub_devices, local_l1_size);
}
SubDeviceManagerId MeshDevice::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return pimpl_->create_sub_device_manager(sub_devices, local_l1_size);
}
void MeshDevice::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    pimpl_->remove_sub_device_manager(sub_device_manager_id);
}
void MeshDevice::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    pimpl_->load_sub_device_manager(sub_device_manager_id);
}
void MeshDevice::clear_loaded_sub_device_manager() { pimpl_->clear_loaded_sub_device_manager(); }
CoreCoord MeshDevice::virtual_program_dispatch_core(uint8_t cq_id) const {
    return pimpl_->virtual_program_dispatch_core(cq_id);
}
const std::vector<SubDeviceId>& MeshDevice::get_sub_device_ids() const { return pimpl_->get_sub_device_ids(); }
const std::vector<SubDeviceId>& MeshDevice::get_sub_device_stall_group() const {
    return pimpl_->get_sub_device_stall_group();
}
void MeshDevice::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    pimpl_->set_sub_device_stall_group(sub_device_ids);
}
void MeshDevice::reset_sub_device_stall_group() { pimpl_->reset_sub_device_stall_group(); }
uint32_t MeshDevice::num_sub_devices() const { return pimpl_->num_sub_devices(); }
bool MeshDevice::is_mmio_capable() const { return pimpl_->is_mmio_capable(); }
std::shared_ptr<distributed::MeshDevice> MeshDevice::get_mesh_device() { return shared_from_this(); }
std::vector<IDevice*> MeshDevice::get_devices() const { return pimpl_->get_devices(); }
IDevice* MeshDevice::get_device(ChipId physical_device_id) const { return pimpl_->get_device(physical_device_id); }
IDevice* MeshDevice::get_device(const MeshCoordinate& coord) const { return pimpl_->get_device(coord); }
tt_fabric::FabricNodeId MeshDevice::get_fabric_node_id(const MeshCoordinate& coord) const {
    return pimpl_->get_fabric_node_id(coord);
}
DeviceIds MeshDevice::get_device_ids() const { return pimpl_->get_device_ids(); }
size_t MeshDevice::num_devices() const { return pimpl_->num_devices(); }
size_t MeshDevice::num_rows() const { return pimpl_->num_rows(); }
size_t MeshDevice::num_cols() const { return pimpl_->num_cols(); }
IDevice* MeshDevice::get_device(size_t row_idx, size_t col_idx) const { return pimpl_->get_device(row_idx, col_idx); }
bool MeshDevice::is_local(const MeshCoordinate& coord) const { return pimpl_->is_local(coord); }
const MeshShape& MeshDevice::shape() const { return pimpl_->shape(); }
void MeshDevice::reshape(const MeshShape& new_shape) { pimpl_->reshape(new_shape); }
const MeshDeviceView& MeshDevice::get_view() const { return pimpl_->get_view(); }
std::string MeshDevice::to_string() const { return pimpl_->to_string(); }
bool MeshDevice::is_parent_mesh() const { return pimpl_->is_parent_mesh(); }
const std::shared_ptr<MeshDevice>& MeshDevice::get_parent_mesh() const { return pimpl_->get_parent_mesh(); }
std::vector<std::shared_ptr<MeshDevice>> MeshDevice::get_submeshes() const { return pimpl_->get_submeshes(); }
void MeshDevice::quiesce_devices() { pimpl_->quiesce_devices(); }
std::shared_ptr<MeshDevice> MeshDevice::create_submesh(
    const MeshShape& submesh_shape, const std::optional<MeshCoordinate>& offset) {
    return pimpl_->create_submesh(shared_from_this(), submesh_shape, offset);
}
std::vector<std::shared_ptr<MeshDevice>> MeshDevice::create_submeshes(const MeshShape& submesh_shape) {
    return pimpl_->create_submeshes(shared_from_this(), submesh_shape);
}
MeshCommandQueue& MeshDevice::mesh_command_queue(std::optional<uint8_t> cq_id) const {
    return pimpl_->mesh_command_queue(cq_id);
}
void MeshDevice::enqueue_to_thread_pool(std::function<void()>&& f) { pimpl_->enqueue_to_thread_pool(std::move(f)); }
void MeshDevice::wait_for_thread_pool() { pimpl_->wait_for_thread_pool(); }

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

}  // namespace tt::tt_metal::distributed
