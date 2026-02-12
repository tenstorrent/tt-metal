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
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
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
#include <tracy/TracyTTDevice.hpp>
#include <common/TracyTTDeviceData.hpp>
#include "tools/profiler/tt_metal_tracy.hpp"
#include <env_lib.hpp>
#include "llrt/hal.hpp"

#include "allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/realtime_profiler_tracy_handler.hpp"
#include "debug/inspector/inspector.hpp"
#include "sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "context/metal_context.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "impl/profiler/profiler_state_manager.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "mesh_device_view_impl.hpp"

namespace tt::tt_metal {
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
        // Catch any exceptions during device close - destructors must not throw.
        // This can happen when a device is hung and times out during close.
        try {
            tt_metal::MetalContext::instance().device_manager()->close_devices(
                devices_to_close, /*skip_synchronize=*/true);
        } catch (const std::exception& e) {
            log_warning(
                LogMetal,
                "Exception during device close in ScopedDevices destructor: {}. "
                "The device may be in an unrecoverable state and require a reset.",
                e.what());
        }
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

    // Initialize D2H socket for real-time profiler streaming
    // This uses the dispatch core which runs dispatch_subordinate kernel
    mesh_device->init_realtime_profiler_socket();

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
    TT_FATAL(
        !device_ids.empty(), "Cannot create unit meshes with empty device_ids. At least one device ID is required.");

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

    // Initialize D2H socket for real-time profiler streaming on each submesh
    // (parent mesh_device is not fully initialized, but each submesh is)
    for (auto& [device_id, submesh] : result) {
        submesh->init_realtime_profiler_socket();
    }

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

    // Stop the real-time profiler receiver thread if running
    if (realtime_profiler_thread_.joinable()) {
        realtime_profiler_stop_.store(true);
        realtime_profiler_thread_.join();
        log_trace(tt::LogMetal, "Real-time profiler receiver thread stopped");
    }

    // Destroy Tracy handler (unregisters callback and cleans up all Tracy contexts)
    realtime_profiler_tracy_handler_.reset();

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

// Sync marker ID - must match device-side REALTIME_PROFILER_SYNC_MARKER_ID
constexpr uint32_t REALTIME_PROFILER_SYNC_MARKER_ID = 0xFFFFFFFF;

void MeshDeviceImpl::run_realtime_profiler_sync(RealtimeProfilerDeviceState& dev_state, uint32_t num_samples) {
    auto& cluster = MetalContext::instance().get_cluster();
    int64_t host_start_time = TracyGetCpuTime();

    struct SyncSample {
        int64_t host_time;     // Full 64-bit host TSC ticks relative to host_start_time
        uint64_t device_time;  // Device wall clock cycles
    };
    std::vector<SyncSample> samples;

    // Enter sync mode - write sync_request = 1
    std::vector<uint32_t> sync_req_data = {1};
    tt::tt_metal::detail::WriteToDeviceL1(
        dev_state.device,
        dev_state.realtime_profiler_core,
        dev_state.sync_request_addr,
        sync_req_data,
        CoreType::WORKER);

    // Collect samples (+1 for the first one we discard)
    for (uint32_t i = 0; i < num_samples + 1; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // Capture host time with full 64-bit precision
        int64_t host_time = TracyGetCpuTime() - host_start_time;

        // Send truncated 32-bit value as echo identifier for pairing
        uint32_t host_time_id = static_cast<uint32_t>(host_time);
        std::vector<uint32_t> host_time_data = {host_time_id};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_host_ts_addr,
            host_time_data,
            CoreType::WORKER);

        // Wait for response from D2H socket
        dev_state.socket->wait_for_pages(1);
        uint32_t* data = dev_state.socket->get_read_ptr();

        // Extract device timestamp and echoed host timestamp
        uint64_t device_time = (static_cast<uint64_t>(data[0]) << 32) | data[1];
        uint32_t echoed_host_time = data[2];
        uint32_t marker = data[3];

        dev_state.socket->pop_pages(1);
        dev_state.socket->notify_sender();

        // Discard first sample - can be very off due to cold PCIe path
        if (i == 0) {
            continue;
        }

        // Verify echo matches and marker is valid; use full 64-bit host_time
        if (marker == REALTIME_PROFILER_SYNC_MARKER_ID && echoed_host_time == host_time_id) {
            samples.push_back({host_time, device_time});
        }
    }

    // Exit sync mode - write sync_request = 0
    sync_req_data[0] = 0;
    tt::tt_metal::detail::WriteToDeviceL1(
        dev_state.device,
        dev_state.realtime_profiler_core,
        dev_state.sync_request_addr,
        sync_req_data,
        CoreType::WORKER);

    // Compute sync parameters using numerically stable centered linear regression.
    //
    // Model: device_time = slope * host_time + intercept
    //   where slope = frequency * tracy_ratio  (device_cycles per TSC_tick)
    //
    // The standard normal equations suffer from catastrophic cancellation when
    // computing (n*Σxy - Σx*Σy) with large absolute values (~10^25) to get a
    // result of ~10^19, losing ~6 digits of precision with double's 15.9 digits.
    //
    // Centering: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    // reduces operand magnitudes from ~10^12 to ~10^9 (deviations from mean),
    // keeping all intermediate products well within double precision.
    if (samples.size() >= 2) {
        const double n = static_cast<double>(samples.size());
        const double tracy_ratio = TracyGetTimerMul();

        // Pass 1: compute means for centering
        double host_mean = 0.0, device_mean = 0.0;
        for (const auto& s : samples) {
            host_mean += static_cast<double>(s.host_time);
            device_mean += static_cast<double>(s.device_time);
        }
        host_mean /= n;
        device_mean /= n;

        // Pass 2: centered regression — numerically stable
        double num = 0.0, den = 0.0;
        for (const auto& s : samples) {
            double dx = static_cast<double>(s.host_time) - host_mean;
            double dy = static_cast<double>(s.device_time) - device_mean;
            num += dx * dy;
            den += dx * dx;
        }

        if (std::abs(den) > 1e-10) {
            // slope = device_cycles per host_TSC_tick
            // frequency = slope / tracy_ratio = device_cycles per nanosecond (GHz)
            double slope = num / den;
            dev_state.sync_frequency = slope / tracy_ratio;
        } else {
            // Fallback to device AICLK
            dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
        }

        // Intercept via means: intercept = ȳ - slope * x̄
        // This is the device cycle count at host_time = 0 (i.e. at host_start_time)
        double slope = dev_state.sync_frequency * tracy_ratio;
        double intercept = device_mean - slope * host_mean;
        dev_state.first_timestamp = static_cast<uint64_t>(intercept);
        dev_state.sync_host_start = host_start_time;

        log_info(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "device_time_at_sync={} cycles",
            dev_state.chip_id,
            samples.size(),
            dev_state.sync_frequency,
            dev_state.first_timestamp);
    } else {
        // Fallback if not enough samples
        dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
        dev_state.first_timestamp = 0;
        dev_state.sync_host_start = host_start_time;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using default frequency",
            dev_state.chip_id);
    }
}

void MeshDeviceImpl::init_realtime_profiler_socket(const std::shared_ptr<MeshDevice>& mesh_device) {
    if (!realtime_profiler_devices_.empty()) {
        return;
    }

    // Real-time profiler is only supported on Blackhole chips due to HW constraints
    // (requires D2H socket support which is not available on Wormhole)
    if (this->arch() != tt::ARCH::BLACKHOLE) {
        log_warning(
            tt::LogMetal,
            "Real-time profiler is not functional on {} due to HW constraints. "
            "Real-time profiler is only supported on Blackhole. Skipping initialization.",
            this->arch());
        return;
    }

    // Configuration for real-time profiler socket
    // Using 64 bytes as minimum PCIe-aligned page size on Blackhole
    constexpr uint32_t kRealtimeProfilerFifoSize = 4096;  // 4KB FIFO for real-time profiler data
    constexpr uint32_t kRealtimeProfilerPageSize = 64;    // 64 bytes per page
    // L1 data buffer - kernel reads address from config
    constexpr uint32_t kL1DataBufferSize = kRealtimeProfilerPageSize;

    // HAL offsets are the same for all devices (same arch)
    const auto& hal = MetalContext::instance().hal();
    const auto& factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    uint32_t realtime_profiler_offset =
        factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::realtime_profiler);
    uint32_t config_buffer_addr_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
        dev_msgs::realtime_profiler_msg_t::Field::config_buffer_addr);
    uint32_t sync_request_offset =
        factory.offset_of<dev_msgs::realtime_profiler_msg_t>(dev_msgs::realtime_profiler_msg_t::Field::sync_request);
    uint32_t sync_host_timestamp_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
        dev_msgs::realtime_profiler_msg_t::Field::sync_host_timestamp);
    uint32_t realtime_profiler_base_addr =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::MAILBOX) + realtime_profiler_offset;
    uint32_t realtime_profiler_mailbox_addr = realtime_profiler_base_addr + config_buffer_addr_offset;

    auto& dispatch_core_manager = MetalContext::instance().get_dispatch_core_manager();
    const std::string realtime_profiler_kernel_path = "tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp";

    // Set up real-time profiler for each local device in the mesh
    for (const auto& coord : MeshCoordinateRange(view_->shape())) {
        if (!this->is_local(coord)) {
            continue;
        }

        IDevice* device = this->get_device(coord);
        auto device_id = device->id();

        // Find the closest available dispatch core to PCIe for real-time profiler
        std::optional<tt_cxy_pair> realtime_profiler_core_opt =
            dispatch_core_manager.get_closest_available_dispatch_core_to_pcie(device_id);

        TT_FATAL(
            realtime_profiler_core_opt.has_value(),
            "No available dispatch core found for real-time profiler on device {}. "
            "Ensure dispatch core descriptor allocates enough cores for dispatch kernels and real-time profiler.",
            device_id);

        CoreCoord realtime_profiler_core(realtime_profiler_core_opt->x, realtime_profiler_core_opt->y);
        log_info(
            tt::LogMetal,
            "Using closest available dispatch core ({}, {}) to PCIe for real-time profiler on device {}",
            realtime_profiler_core.x,
            realtime_profiler_core.y,
            device_id);

        // Create per-device state
        RealtimeProfilerDeviceState dev_state;
        dev_state.device = device;
        dev_state.chip_id = device_id;
        dev_state.mesh_coord = coord;
        dev_state.realtime_profiler_core = realtime_profiler_core;

        // Create D2H socket for this device
        auto sender_core = MeshCoreCoord{coord, realtime_profiler_core};

        log_info(
            tt::LogMetal,
            "Initializing real-time profiler D2H socket for device {} on MeshDevice {}",
            device_id,
            this->id());

        dev_state.socket = std::make_unique<D2HSocket>(
            mesh_device, sender_core, BufferType::L1, kRealtimeProfilerFifoSize, kL1DataBufferSize);
        dev_state.socket->set_page_size(kRealtimeProfilerPageSize);

        // Populate L1 data buffer with test data
        uint32_t l1_data_addr = dev_state.socket->get_l1_data_buffer_address();
        if (l1_data_addr != 0) {
            std::vector<uint32_t> test_data(kRealtimeProfilerPageSize / sizeof(uint32_t));
            for (uint32_t i = 0; i < test_data.size(); i++) {
                test_data[i] = 0xDEAD0000 + i;
            }
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, l1_data_addr, test_data, CoreType::WORKER);
        }

        // Write config buffer address to mailbox
        uint32_t config_buffer_addr = dev_state.socket->get_config_buffer_address();
        std::vector<uint32_t> addr_data = {config_buffer_addr};
        tt::tt_metal::detail::WriteToDeviceL1(
            device, realtime_profiler_core, realtime_profiler_mailbox_addr, addr_data, CoreType::WORKER);

        log_info(
            tt::LogMetal,
            "Device {}: wrote real-time profiler config buffer addr 0x{:x} to mailbox addr 0x{:x}",
            device_id,
            config_buffer_addr,
            realtime_profiler_mailbox_addr);

        // Store sync addresses
        dev_state.sync_request_addr = realtime_profiler_base_addr + sync_request_offset;
        dev_state.sync_host_ts_addr = realtime_profiler_base_addr + sync_host_timestamp_offset;

        // Write real-time profiler core info to dispatch_s core's mailbox for termination signaling
        if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
            const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, 0, 0);
            CoreCoord dispatch_s_core(dispatch_s_cxy.x, dispatch_s_cxy.y);

            CoreCoord realtime_profiler_virtual =
                device->virtual_core_from_logical_core(realtime_profiler_core, CoreType::WORKER);
            uint32_t realtime_profiler_noc_xy =
                hal.noc_xy_encoding(realtime_profiler_virtual.x, realtime_profiler_virtual.y);

            uint32_t realtime_profiler_core_noc_xy_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
                dev_msgs::realtime_profiler_msg_t::Field::realtime_profiler_core_noc_xy);
            uint32_t realtime_profiler_mailbox_addr_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
                dev_msgs::realtime_profiler_msg_t::Field::realtime_profiler_mailbox_addr);
            uint32_t realtime_profiler_state_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
                dev_msgs::realtime_profiler_msg_t::Field::realtime_profiler_state);
            uint32_t realtime_profiler_core_state_addr = realtime_profiler_base_addr + realtime_profiler_state_offset;
            uint32_t dispatch_s_mailbox_base = realtime_profiler_base_addr;

            std::vector<uint32_t> noc_xy_data = {realtime_profiler_noc_xy};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                dispatch_s_mailbox_base + realtime_profiler_core_noc_xy_offset,
                noc_xy_data,
                CoreType::WORKER);

            std::vector<uint32_t> mailbox_addr_data = {realtime_profiler_core_state_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                dispatch_s_mailbox_base + realtime_profiler_mailbox_addr_offset,
                mailbox_addr_data,
                CoreType::WORKER);

            log_info(
                tt::LogMetal,
                "Device {}: wrote real-time profiler core info (noc_xy=0x{:x}, mailbox_addr=0x{:x}) to dispatch_s ({}, "
                "{})",
                device_id,
                realtime_profiler_noc_xy,
                realtime_profiler_core_state_addr,
                dispatch_s_core.x,
                dispatch_s_core.y);
        }

        // Compile and launch real-time profiler kernel via slow dispatch
        {
            Program realtime_profiler_program;

            uint32_t dispatch_core_noc_x = 0;
            uint32_t dispatch_core_noc_y = 0;
            uint32_t dispatch_data_addr_a = 0;
            uint32_t dispatch_data_addr_b = 0;
            if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
                const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, 0, 0);
                CoreCoord dispatch_s_virtual = device->virtual_core_from_logical_core(
                    CoreCoord(dispatch_s_cxy.x, dispatch_s_cxy.y), CoreType::WORKER);
                dispatch_core_noc_x = dispatch_s_virtual.x;
                dispatch_core_noc_y = dispatch_s_virtual.y;

                uint32_t kernel_start_a_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
                    dev_msgs::realtime_profiler_msg_t::Field::kernel_start_a);
                uint32_t kernel_start_b_offset = factory.offset_of<dev_msgs::realtime_profiler_msg_t>(
                    dev_msgs::realtime_profiler_msg_t::Field::kernel_start_b);
                dispatch_data_addr_a = realtime_profiler_base_addr + kernel_start_a_offset;
                dispatch_data_addr_b = realtime_profiler_base_addr + kernel_start_b_offset;
            }

            DataMovementConfig realtime_profiler_config;
            realtime_profiler_config.defines["DISPATCH_CORE_NOC_X"] = std::to_string(dispatch_core_noc_x);
            realtime_profiler_config.defines["DISPATCH_CORE_NOC_Y"] = std::to_string(dispatch_core_noc_y);
            realtime_profiler_config.defines["DISPATCH_DATA_ADDR_A"] = std::to_string(dispatch_data_addr_a);
            realtime_profiler_config.defines["DISPATCH_DATA_ADDR_B"] = std::to_string(dispatch_data_addr_b);
            CreateKernel(
                realtime_profiler_program,
                realtime_profiler_kernel_path,
                realtime_profiler_core,
                realtime_profiler_config);

            tt::tt_metal::detail::CompileProgram(device, realtime_profiler_program, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::detail::WriteRuntimeArgsToDevice(
                device, realtime_profiler_program, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::detail::LaunchProgram(
                device, realtime_profiler_program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

            log_info(
                tt::LogMetal,
                "Device {}: launched real-time profiler kernel on core ({}, {})",
                device_id,
                realtime_profiler_core.x,
                realtime_profiler_core.y);
        }

        realtime_profiler_devices_.push_back(std::move(dev_state));
    }

    if (realtime_profiler_devices_.empty()) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] No local devices found in mesh, skipping real-time profiler setup");
        return;
    }

    // Run host-device sync for each device, or use profiler sync value if already available
    const std::unique_ptr<tt::tt_metal::ProfilerStateManager>& profiler_state_manager =
        MetalContext::instance().profiler_state_manager();
    const bool use_profiler_sync =
        MetalContext::instance().rtoptions().get_profiler_sync_enabled() && (profiler_state_manager != nullptr);

    for (auto& dev_state : realtime_profiler_devices_) {
        std::optional<tt::tt_metal::SyncInfo> profiler_sync_info;
        if (use_profiler_sync) {
            auto it = profiler_state_manager->device_profiler_map.find(dev_state.chip_id);
            if (it != profiler_state_manager->device_profiler_map.end()) {
                const tt::tt_metal::DeviceProfiler& device_profiler = it->second;
                if (!device_profiler.device_core_sync_info.empty()) {
                    profiler_sync_info = device_profiler.device_core_sync_info.begin()->second;
                }
            }
        }

        if (profiler_sync_info.has_value()) {
            const tt::tt_metal::SyncInfo& sync_info = profiler_sync_info.value();
            dev_state.sync_host_start = static_cast<int64_t>(sync_info.cpu_time);
            dev_state.first_timestamp = static_cast<uint64_t>(sync_info.device_time);
            // Realtime profiler expects frequency in GHz (cycles per ns). Profiler sync stores frequency
            // in Hz (cycles per second) from the regression; convert to GHz. If value is already in GHz
            // (e.g. < 1000), use as-is.
            dev_state.sync_frequency =
                (sync_info.frequency >= 1000.0) ? (sync_info.frequency / 1e9) : sync_info.frequency;
            log_info(
                tt::LogMetal,
                "[Real-time profiler] Device {} using profiler sync: frequency={:.6f} GHz, device_time_at_sync={} "
                "cycles",
                dev_state.chip_id,
                dev_state.sync_frequency,
                dev_state.first_timestamp);
        } else {
            run_realtime_profiler_sync(dev_state, 100);  // 100 samples, discard first
        }
    }

    // Create Tracy handler and register all device contexts
    realtime_profiler_tracy_handler_ = std::make_unique<RealtimeProfilerTracyHandler>();
    for (const auto& dev_state : realtime_profiler_devices_) {
        realtime_profiler_tracy_handler_->AddDevice(
            dev_state.chip_id,
            dev_state.sync_host_start,
            static_cast<double>(dev_state.first_timestamp),
            dev_state.sync_frequency);
    }

    // Start background receiver thread that polls all device sockets round-robin
    realtime_profiler_stop_.store(false);
    realtime_profiler_thread_ = std::thread([this]() {
        uint64_t pages_received = 0;

        log_info(
            tt::LogMetal,
            "[Real-time profiler] Receiver thread started for {} devices",
            realtime_profiler_devices_.size());

        while (!realtime_profiler_stop_.load()) {
            bool any_data = false;

            for (auto& dev_state : realtime_profiler_devices_) {
                try {
                    uint32_t available = dev_state.socket->pages_available();
                    if (available == 0) {
                        continue;
                    }

                    dev_state.socket->wait_for_pages(1);
                    uint32_t* read_ptr = dev_state.socket->get_read_ptr();

                    // Extract timestamps: kernel_start (words 0-3), kernel_end (words 4-7)
                    // Each realtime_profiler_timestamp_t: time_hi, time_lo, id, header
                    uint64_t start_time = (static_cast<uint64_t>(read_ptr[0]) << 32) | read_ptr[1];
                    uint32_t start_id = read_ptr[2];
                    uint64_t end_time = (static_cast<uint64_t>(read_ptr[4]) << 32) | read_ptr[5];

                    // Invoke registered real-time profiler callbacks.
                    // Skip records with id==0: these are non-GO dispatch commands
                    // (e.g. SET_NUM_WORKER_SEMS, SET_GO_SIGNAL_NOC_DATA) that have
                    // no valid program and may contain stale end timestamps.
                    if (start_id != 0) {
                        tt::ProgramRealtimeRecord record;
                        record.program_id = start_id;
                        record.start_timestamp = start_time;
                        record.end_timestamp = end_time;
                        record.frequency = dev_state.sync_frequency;
                        record.chip_id = dev_state.chip_id;
                        record.kernel_sources = tt::GetKernelSourcesVecForRuntimeId(start_id);
                        tt::InvokeProgramRealtimeProfilerCallbacks(record);
                    }

                    dev_state.socket->pop_pages(1);
                    dev_state.socket->notify_sender();
                    pages_received++;
                    any_data = true;
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Exception in receiver for device {}: {}",
                        dev_state.chip_id,
                        e.what());
                }
            }

            if (!any_data) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        log_info(tt::LogMetal, "[Real-time profiler] Receiver thread stopped after {} pages", pages_received);
    });
}

D2HSocket* MeshDeviceImpl::get_realtime_profiler_socket() const {
    return realtime_profiler_devices_.empty() ? nullptr : realtime_profiler_devices_.front().socket.get();
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
    TT_FATAL(
        sub_device_manager_tracker_ != nullptr, "sub_device_manager_tracker_ is NULL! MeshDeviceImpl not initialized.");
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
void MeshDevice::init_realtime_profiler_socket() { pimpl_->init_realtime_profiler_socket(shared_from_this()); }
D2HSocket* MeshDevice::get_realtime_profiler_socket() const { return pimpl_->get_realtime_profiler_socket(); }
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
