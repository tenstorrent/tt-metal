// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <tt-logger/tt-logger.hpp>
#include <mesh_command_queue.hpp>
#include <mesh_coord.hpp>
#include <mesh_device.hpp>
#include <mesh_device_view.hpp>
#include <tt_stl/small_vector.hpp>
#include <sub_device.hpp>
#include <system_mesh.hpp>
#include <maybe_remote.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <source_location>
#include <unordered_map>
#include <utility>

#include "allocator.hpp"
#include "assert.hpp"
#include "buffer.hpp"
#include "device/device_impl.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "host_api.hpp"
#include "mesh_trace.hpp"
#include "profiler_types.hpp"
#include "shape_base.hpp"
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/api/tt-metalium/device_pool.hpp"
#include "tt_metal/api/tt-metalium/control_plane.hpp"
#include "tt_metal/api/tt-metalium/fabric_types.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/distributed/sd_mesh_command_queue.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/tt_metal_tracy.hpp"
#include "tt_metal/distributed/distributed_coordinate_system.hpp"

#include "llrt/hal.hpp"
#include <env_lib.hpp>

#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/debug/inspector.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include <umd/device/types/xy_pair.h>
#include "impl/context/metal_context.hpp"

enum class CoreType;
namespace tt {
namespace tt_metal {
class CommandQueue;
class SystemMemoryManager;
namespace program_cache {
namespace detail {
struct ProgramCache;
}  // namespace detail
}  // namespace program_cache
}  // namespace tt_metal
}  // namespace tt

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
    } else if (tt::parse_env("TT_MESH_BOOST_THREAD_POOL", false)) {
        return create_boost_thread_pool(physical_devices.size());
    } else {
        return create_device_bound_thread_pool(physical_devices);
    }
}

// Helper function to verify all devices in the MeshDevice have the same value
template <typename F>
decltype(auto) validate_and_get_reference_value(
    const std::vector<IDevice*>& devices, F&& func, const std::source_location& loc = std::source_location::current()) {
    if (devices.empty()) {
        TT_THROW("{} [{}:{}] failed: MeshDevice has no devices", loc.function_name(), loc.file_name(), loc.line());
    }

    // Get reference to first device's value
    decltype(auto) reference_value = std::forward<F>(func)(devices.front());

    // Validate all other devices match
    for (auto it = devices.begin() + 1; it != devices.end(); ++it) {
        const auto& current_value = std::forward<F>(func)(*it);
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

}  // namespace

MeshDevice::ScopedDevices::ScopedDevices(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    size_t worker_l1_size,
    const DispatchCoreConfig& dispatch_core_config,
    const MeshDeviceConfig& config) :
    ScopedDevices(
        config.physical_device_ids().empty()
            ? SystemMesh::instance().get_mapped_physical_device_ids(config.mesh_shape(), config.offset())
            : config.physical_device_ids(),
        l1_small_size,
        trace_region_size,
        num_command_queues,
        worker_l1_size,
        dispatch_core_config) {}

MeshDevice::ScopedDevices::ScopedDevices(
    const std::vector<int>& device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    size_t worker_l1_size,
    const DispatchCoreConfig& dispatch_core_config) {
    opened_devices_ = tt::tt_metal::detail::CreateDevices(
        device_ids,
        num_command_queues,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        {},
        worker_l1_size,
        /*init_profiler*/ false,
        /*use_max_eth_core_count_on_all_devices*/ true,
        /* initialize_fabric_and_dispatch_fw */ false);

    for (auto device_id : device_ids) {
        devices_.push_back(opened_devices_.at(device_id));
    }
}

MeshDevice::ScopedDevices::~ScopedDevices() {
    if (!opened_devices_.empty()) {
        std::vector<IDevice*> devices_to_close;
        devices_to_close.reserve(opened_devices_.size());
        for (auto& [id, device] : opened_devices_) {
            devices_to_close.push_back(device);
        }
        tt::DevicePool::instance().close_devices(devices_to_close, /*skip_synchronize=*/true);
    }
}

const std::vector<IDevice*>& MeshDevice::ScopedDevices::root_devices() const { return devices_; }

uint8_t MeshDevice::num_hw_cqs() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->num_hw_cqs(); });
}

bool MeshDevice::is_initialized() const {
    if (!scoped_devices_) {
        return false;
    }
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->is_initialized(); });
}

uint32_t MeshDevice::l1_size_per_core() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->l1_size_per_core(); });
}

uint32_t MeshDevice::dram_size_per_channel() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->dram_size_per_channel(); });
}

IDevice* MeshDevice::reference_device() const { return this->get_devices().at(0); }

MeshDevice::MeshDevice(
    std::shared_ptr<ScopedDevices> mesh_handle,
    std::unique_ptr<MeshDeviceView> mesh_device_view,
    std::shared_ptr<MeshDevice> parent_mesh) :
    scoped_devices_(std::move(mesh_handle)),
    view_(std::move(mesh_device_view)),
    mesh_id_(generate_unique_mesh_id()),
    parent_mesh_(std::move(parent_mesh)),
    program_cache_(std::make_unique<program_cache::detail::ProgramCache>()),
    dispatch_thread_pool_(create_default_thread_pool(scoped_devices_->root_devices())),
    reader_thread_pool_(create_default_thread_pool(scoped_devices_->root_devices())) {
    Inspector::mesh_device_created(this, parent_mesh_ ? std::make_optional(parent_mesh_->mesh_id_) : std::nullopt);
}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    auto scoped_devices = std::make_shared<ScopedDevices>(
        l1_small_size, trace_region_size, num_command_queues, worker_l1_size, dispatch_core_config, config);
    auto root_devices = scoped_devices->root_devices();

    // When the mesh is distributed across multiple ranks, the DistributedCoordinateSystem
    // is used to provide information on what part of the mesh is locally available vs. remote.
    // For single-process/ single-host, the "global" mesh is the same as the "local" mesh.
    auto coord_system = DistributedCoordinateSystem::from_control_plane();

    // Create distributed mesh container and populate local devices
    DistributedMeshContainer<IDevice*> global_devices(config.mesh_shape());
    global_devices.populate_local_region(coord_system, root_devices);

    auto mesh_device = std::make_shared<MeshDevice>(
        std::move(scoped_devices),
        std::make_unique<MeshDeviceView>(global_devices),
        std::shared_ptr<MeshDevice>());

    mesh_device->initialize(num_command_queues, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap);
    // TODO #20966: Remove these calls
    for (auto device : root_devices) {
        dynamic_cast<Device*>(device)->set_mesh_device(mesh_device);
    }
    // The Device Profiler must be initialized before Fabric is loaded on the Cluster
    DevicePool::instance().init_profiler();
    DevicePool::instance().initialize_fabric_and_dispatch_fw();
    return mesh_device;
}

void MeshDevice::enqueue_to_thread_pool(std::function<void()>&& f) { dispatch_thread_pool_->enqueue(std::move(f)); }

void MeshDevice::wait_for_thread_pool() { dispatch_thread_pool_->wait(); }

std::map<int, std::shared_ptr<MeshDevice>> MeshDevice::create_unit_meshes(
    const std::vector<int>& device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size) {
    auto scoped_devices = std::make_shared<ScopedDevices>(
        device_ids, l1_small_size, trace_region_size, num_command_queues, worker_l1_size, dispatch_core_config);
    MeshContainer<IDevice*> devices(MeshShape(1, device_ids.size()), scoped_devices->root_devices());
    auto mesh_device = std::make_shared<MeshDevice>(
        std::move(scoped_devices), std::make_unique<MeshDeviceView>(devices), std::shared_ptr<MeshDevice>());

    auto submeshes = mesh_device->create_submeshes(MeshShape(1, 1));
    TT_FATAL(
        device_ids.size() == submeshes.size(),
        "Created an unexpected number of submeshes: {} instead of {}",
        submeshes.size(),
        device_ids.size());
    std::map<int, std::shared_ptr<MeshDevice>> result;
    for (size_t i = 0; i < device_ids.size(); i++) {
        submeshes[i]->initialize(num_command_queues, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap);
        result[device_ids[i]] = submeshes[i];
    }
    // The Device Profiler must be initialized before Fabric is loaded on the Cluster
    DevicePool::instance().init_profiler();
    DevicePool::instance().initialize_fabric_and_dispatch_fw();
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

std::shared_ptr<MeshDevice> MeshDevice::create_submesh(
    const MeshShape& submesh_shape, const std::optional<MeshCoordinate>& offset) {
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
        } else {
            return MeshCoordinate::zero_coordinate(submesh_shape.dims());
        }
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

    MeshContainer<IDevice*> submesh_devices_container(
        submesh_shape, view_->get_devices(MeshCoordinateRange{offset_coord, end_coordinate}));

    auto submesh = std::make_shared<MeshDevice>(
        scoped_devices_, std::make_unique<MeshDeviceView>(submesh_devices_container), shared_from_this());
    const auto& allocator_config = reference_device()->allocator()->get_config();
    submesh->initialize(
        num_hw_cqs(),
        allocator_config.l1_small_size,
        allocator_config.trace_region_size,
        allocator_config.worker_l1_size,
        allocator_config.l1_bank_remap);
    // TODO #20966: Remove these calls
    for (auto device : submesh->get_devices()) {
        dynamic_cast<Device*>(device)->set_mesh_device(submesh);
    }

    submeshes_.push_back(submesh);
    log_trace(LogMetal, "Instantiating submesh {}: {} with offset: {}", submesh->id(), submesh_shape, offset);
    log_trace(LogMetal, "Submesh {} instantiated with {} devices", submesh->id(), submesh->get_devices().size());
    return submesh;
}

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::create_submeshes(const MeshShape& submesh_shape) {
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
        submeshes.push_back(create_submesh(submesh_shape, MeshCoordinate(offset_coords)));
    }

    return submeshes;
}

MeshDevice::~MeshDevice() {
    Inspector::mesh_device_destroyed(this);
    close();
}

IDevice* MeshDevice::get_device(chip_id_t physical_device_id) const {
    for (auto device : this->get_devices()) {
        if (device->id() == physical_device_id) {
            return device;
        }
    }
    TT_THROW("Physical Device ID: {} not found in assigned devices", physical_device_id);
}

std::vector<IDevice*> MeshDevice::get_devices() const { return view_->get_devices(); }

// TODO: Remove this function once we have a proper view interface
IDevice* MeshDevice::get_device(size_t row_idx, size_t col_idx) const {
    return get_device(MeshCoordinate{row_idx, col_idx});
}

IDevice* MeshDevice::get_device(const MeshCoordinate& coord) const { return view_->get_device(coord); }

tt_fabric::FabricNodeId MeshDevice::get_device_fabric_node_id(const MeshCoordinate& coord) const {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_node_id_from_physical_chip_id(view_->get_device(coord)->id());
}

MeshCommandQueue& MeshDevice::mesh_command_queue(std::size_t cq_id) const {
    TT_FATAL(cq_id < mesh_command_queues_.size(), "cq_id {} is out of range", cq_id);
    return *(mesh_command_queues_[cq_id]);
}

DeviceIds MeshDevice::get_device_ids() const {
    DeviceIds device_ids;
    for (auto device : this->get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

size_t MeshDevice::num_devices() const { return view_->num_devices(); }

CoreCoord MeshDevice::compute_with_storage_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->compute_with_storage_grid_size(); });
}

tt::ARCH MeshDevice::arch() const {
    return validate_and_get_reference_value(this->get_devices(), [](const auto* device) { return device->arch(); });
}

size_t MeshDevice::num_rows() const { return view_->num_rows(); }

size_t MeshDevice::num_cols() const { return view_->num_cols(); }

const MeshShape& MeshDevice::shape() const { return view_->shape(); }

std::vector<IDevice*> MeshDevice::get_row_major_devices(const MeshShape& new_shape) const {
    TT_FATAL(
        this->shape() == this->local_shape(), "Cannot reshape a mesh that is not the same shape as the local shape");

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
    std::unordered_map<chip_id_t, size_t> physical_device_id_to_linearized_index;
    for (size_t i = 0; i < this->num_devices(); i++) {
        physical_device_id_to_linearized_index[this->get_devices()[i]->id()] = i;
    }

    // From an MxN mesh, we can always reduce rank to a 1xM*N Line mesh.
    // However, going from a Line mesh to an MxN mesh is not always possible.
    if (new_shape.is_line_topology()) {
        return view_->get_line_devices();
    }

    auto new_physical_device_ids = SystemMesh::instance().get_mapped_physical_device_ids(new_shape);

    for (size_t i = 0; i < new_physical_device_ids.size(); i++) {
        if (physical_device_id_to_linearized_index.find(new_physical_device_ids[i]) ==
            physical_device_id_to_linearized_index.end()) {
            TT_THROW(
                "User has requested a reshape of the MeshDevice to shape: {}, but it is not possible to form a "
                "physically connected mesh grid with the opened devices from the original shape: {}.",
                new_shape,
                view_->shape());
        }
    }

    std::vector<IDevice*> new_device_order;
    for (size_t i = 0; i < new_physical_device_ids.size(); i++) {
        new_device_order.push_back(this->get_device(new_physical_device_ids[i]));
    }
    return new_device_order;
}

void MeshDevice::reshape(const MeshShape& new_shape) {
    TT_FATAL(
        this->shape() == this->local_shape(),
        "Cannot reshape a mesh that is not the same shape as the local shape");

    TT_FATAL(
        new_shape.mesh_size() == this->num_devices(),
        "New shape must have the same number of devices as current shape");

    MeshContainer<IDevice*> devices(new_shape, this->get_row_major_devices(new_shape));
    auto new_view = std::make_unique<MeshDeviceView>(devices);
    view_ = std::move(new_view);
}

bool MeshDevice::close() {
    ZoneScoped;
    log_trace(tt::LogMetal, "Closing mesh device {}", this->id());

    // We only dump profile results for mesh devices that don't have any submeshes as they have active mesh command
    // queues, whereas mesh devices with submeshes don't.
    if (this->submeshes_.empty()) {
        DumpMeshDeviceProfileResults(*this);
    }

    // TODO #20966: Remove these calls
    for (auto device : view_->get_devices()) {
        dynamic_cast<Device*>(device)->set_mesh_device(parent_mesh_);
    }

    mesh_command_queues_.clear();
    sub_device_manager_tracker_.reset();
    scoped_devices_.reset();
    parent_mesh_.reset();
    return true;
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", this->num_rows(), this->num_cols(), this->num_devices());
}

const MeshDeviceView& MeshDevice::get_view() const {
    TT_FATAL(view_, "MeshDeviceView is not initialized");
    return *view_;
}

int MeshDevice::id() const { return mesh_id_; }
// For a mesh, build id is the same as the device id for the reference device
chip_id_t MeshDevice::build_id() const { return reference_device()->id(); }

bool MeshDevice::is_parent_mesh() const { return parent_mesh_ == nullptr; }

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::get_submeshes() const {
    std::vector<std::shared_ptr<MeshDevice>> result;
    result.reserve(submeshes_.size());
    for (const auto& weak_submesh : submeshes_) {
        if (auto submesh = weak_submesh.lock()) {
            result.push_back(std::move(submesh));
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

void MeshDevice::enable_program_cache() {
    log_info(tt::LogMetal, "Enabling program cache on MeshDevice {}", this->id());
    program_cache_->enable();
}

void MeshDevice::clear_program_cache() {
    log_info(tt::LogMetal, "Clearing program cache on MeshDevice {}", this->id());
    program_cache_->clear();
}

void MeshDevice::disable_and_clear_program_cache() {
    log_info(tt::LogMetal, "Disabling and clearing program cache on MeshDevice {}", this->id());
    if (program_cache_->is_enabled()) {
        program_cache_->disable();
    }
    program_cache_->clear();
}

size_t MeshDevice::num_program_cache_entries() { return program_cache_->num_entries(); }

SubDeviceManagerId MeshDevice::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}
void MeshDevice::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->remove_sub_device_manager(sub_device_manager_id);
}
void MeshDevice::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->load_sub_device_manager(sub_device_manager_id);
}
void MeshDevice::clear_loaded_sub_device_manager() { sub_device_manager_tracker_->clear_loaded_sub_device_manager(); }

CoreCoord MeshDevice::dram_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->dram_grid_size(); });
}

bool MeshDevice::using_slow_dispatch() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->using_slow_dispatch(); });
}

bool MeshDevice::using_fast_dispatch() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->using_fast_dispatch(); });
}

// Device property methods that can be delegated to reference device
CoreCoord MeshDevice::grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->grid_size(); });
}
CoreCoord MeshDevice::logical_grid_size() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) { return device->logical_grid_size(); });
}
CoreCoord MeshDevice::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    return validate_and_get_reference_value(this->get_devices(), [noc_index, coord](const auto* device) {
        return device->virtual_noc0_coordinate(noc_index, coord);
    });
}
std::vector<CoreCoord> MeshDevice::worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_cores](const auto* device) {
        return device->worker_cores_from_logical_cores(logical_cores);
    });
}
std::vector<CoreCoord> MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) {
    return validate_and_get_reference_value(this->get_devices(), [noc](auto* device) {
        return device->get_optimal_dram_bank_to_logical_worker_assignment(noc);
    });
}
CoreCoord MeshDevice::virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_coord, core_type](const auto* device) {
        return device->virtual_core_from_logical_core(logical_coord, core_type);
    });
}
CoreCoord MeshDevice::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->worker_core_from_logical_core(logical_core);
    });
}
CoreCoord MeshDevice::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return validate_and_get_reference_value(this->get_devices(), [ethernet_core](const auto* device) {
        return device->logical_core_from_ethernet_core(ethernet_core);
    });
}

// These methods require some change / or assert out for now
std::vector<CoreCoord> MeshDevice::ethernet_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_cores](const auto* device) {
        return device->ethernet_cores_from_logical_cores(logical_cores);
    });
}
CoreCoord MeshDevice::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->ethernet_core_from_logical_core(logical_core);
    });
}
std::unordered_set<CoreCoord> MeshDevice::get_active_ethernet_cores(bool /*skip_reserved_tunnel_cores*/) const {
    TT_THROW("get_active_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

std::unordered_set<CoreCoord> MeshDevice::get_inactive_ethernet_cores() const {
    TT_THROW("get_inactive_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDevice::is_inactive_ethernet_core(CoreCoord /*logical_core*/) const {
    TT_THROW("is_inactive_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::tuple<chip_id_t, CoreCoord> MeshDevice::get_connected_ethernet_core(CoreCoord /*eth_core*/) const {
    TT_THROW("get_connected_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDevice::is_active_ethernet_core(CoreCoord /*logical_core*/, bool /*skip_reserved_tunnel_cores*/) const {
    TT_THROW("is_active_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::vector<CoreCoord> MeshDevice::get_ethernet_sockets(chip_id_t /*connected_chip_id*/) const {
    TT_THROW("get_ethernet_sockets() is not supported on MeshDevice - use individual devices instead");
}

uint32_t MeshDevice::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    // Issue #19729: Return the maximum number of active ethernet cores across physical devices in the Mesh.
    TT_FATAL(*sub_device_id == 0, "Cannot query virtual ethernet cores per sub-device when using MeshDevice");
    return num_virtual_eth_cores_;
}

// Core and worker management methods (These are OK)
CoreRangeSet MeshDevice::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}
uint32_t MeshDevice::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).num_cores(core_type);
}

// Bank and memory management methods
int MeshDevice::num_dram_channels() const { return reference_device()->num_dram_channels(); }

CoreCoord MeshDevice::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return validate_and_get_reference_value(this->get_devices(), [dram_channel](const auto* device) {
        return device->logical_core_from_dram_channel(dram_channel);
    });
}
uint32_t MeshDevice::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return validate_and_get_reference_value(this->get_devices(), [logical_core](const auto* device) {
        return device->dram_channel_from_logical_core(logical_core);
    });
}
uint32_t MeshDevice::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    return validate_and_get_reference_value(this->get_devices(), [virtual_core](const auto* device) {
        return device->dram_channel_from_virtual_core(virtual_core);
    });
}

// Core management and network operations
const std::set<CoreCoord>& MeshDevice::ethernet_cores() const {
    return validate_and_get_reference_value(
        this->get_devices(), [](const auto* device) -> const std::set<CoreCoord>& { return device->ethernet_cores(); });
}
const std::set<CoreCoord>& MeshDevice::storage_only_cores() const {
    return validate_and_get_reference_value(this->get_devices(), [](const auto* device) -> const std::set<CoreCoord>& {
        return device->storage_only_cores();
    });
}
uint32_t MeshDevice::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    return validate_and_get_reference_value(this->get_devices(), [noc_index, core](const auto* device) {
        return device->get_noc_unicast_encoding(noc_index, core);
    });
}
uint32_t MeshDevice::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    return validate_and_get_reference_value(this->get_devices(), [noc_index, cores](const auto* device) {
        return device->get_noc_multicast_encoding(noc_index, cores);
    });
}

// System memory and command queue management
SystemMemoryManager& MeshDevice::sysmem_manager() {
    TT_THROW("sysmem_manager() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->sysmem_manager();
}

CommandQueue& MeshDevice::command_queue(size_t cq_id) {
    TT_THROW("command_queue() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->command_queue(cq_id);
}

// Trace management
void MeshDevice::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_THROW("begin_trace() is not supported on MeshDevice");
}
void MeshDevice::end_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_THROW("end_trace() is not supported on MeshDevice");
}
void MeshDevice::replay_trace(
    const uint8_t cq_id, const uint32_t tid, const bool block_on_device, const bool block_on_worker_thread) {
    TT_THROW("replay_trace() is not supported on MeshDevice");
}

void MeshDevice::release_trace(const uint32_t tid) { TT_THROW("release_trace() is not supported on MeshDevice"); }

std::shared_ptr<MeshTraceBuffer>& MeshDevice::create_mesh_trace(const MeshTraceId& trace_id) {
    auto [trace, emplaced] = trace_buffer_pool_.emplace(trace_id, MeshTrace::create_empty_mesh_trace_buffer());
    TT_FATAL(emplaced, "Trace buffer with tid {} already exists", *trace_id);
    return trace->second;
}

void MeshDevice::release_mesh_trace(const MeshTraceId& trace_id) {
    TracyTTMetalReleaseMeshTrace(this->get_device_ids(), *trace_id);
    const auto& trace_mesh_buffer = trace_buffer_pool_.at(trace_id)->mesh_buffer;
    TT_FATAL(
        trace_mesh_buffer and trace_mesh_buffer->is_allocated(),
        "Trace buffer for {} is not allocated when calling {}",
        *trace_id,
        __FUNCTION__);
    auto current_trace_buffers_size = this->get_trace_buffers_size();
    this->set_trace_buffers_size(current_trace_buffers_size - trace_mesh_buffer->size());
    trace_buffer_pool_.erase(trace_id);
}

std::shared_ptr<MeshTraceBuffer> MeshDevice::get_mesh_trace(const MeshTraceId& trace_id) {
    auto trace = trace_buffer_pool_.find(trace_id);
    if (trace != trace_buffer_pool_.end()) {
        return trace->second;
    }
    TT_THROW("MeshDevice ID {} Trace Instance with ID {} is not initialized", this->id(), *trace_id);
}

void MeshDevice::begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalBeginMeshTrace(this->get_device_ids(), *trace_id);
    auto& mesh_trace_buffer = this->create_mesh_trace(trace_id);
    mesh_command_queues_[cq_id]->record_begin(trace_id, mesh_trace_buffer->desc);
}

void MeshDevice::end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalEndMeshTrace(this->get_device_ids(), *trace_id);
    auto trace_buffer = this->get_mesh_trace(trace_id);
    mesh_command_queues_[cq_id]->record_end();
    MeshTrace::populate_mesh_buffer(*(mesh_command_queues_[cq_id]), trace_buffer);
}

void MeshDevice::replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    TracyTTMetalReplayMeshTrace(this->get_device_ids(), *trace_id);
    mesh_command_queues_[cq_id]->enqueue_trace(trace_id, blocking);
}

std::shared_ptr<TraceBuffer> MeshDevice::get_trace(uint32_t tid) {
    TT_THROW("get_trace() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_trace(tid);
}
uint32_t MeshDevice::get_trace_buffers_size() const { return trace_buffers_size_; }
void MeshDevice::set_trace_buffers_size(uint32_t size) { trace_buffers_size_ = size; }

// Light Metal
void MeshDevice::load_trace(const uint8_t cq_id, const uint32_t trace_id, const TraceDescriptor& trace_desc) {
    TT_THROW("load_trace() is not supported on MeshDevice - use individual devices instead");
    reference_device()->load_trace(cq_id, trace_id, trace_desc);
}

// Dispatch and initialization
bool MeshDevice::initialize(
    const uint8_t /*num_hw_cqs*/,
    size_t /*l1_small_size*/,
    size_t /*trace_region_size*/,
    size_t /*worker_l1_size*/,
    tt::stl::Span<const std::uint32_t> /*l1_bank_remap*/,
    bool /*minimal*/) {
    // For MeshDevice, we support uniform sub-devices across all devices and we do not support ethernet subdevices.
    const auto& compute_grid_size = this->compute_with_storage_grid_size();
    auto sub_devices = {
        SubDevice(std::array{CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}))})};

    // Resource shared across mesh command queues.
    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);

    const auto& allocator = reference_device()->allocator();
    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        this, std::make_unique<L1BankingAllocator>(allocator->get_config()), sub_devices);
    // Issue #19729: Store the maximum number of active ethernet cores across opened physical devices in the Mesh
    // as the number of virtual ethernet cores seen by the MeshDevice
    num_virtual_eth_cores_ = DevicePool::instance().get_max_num_eth_cores_across_all_devices();
    mesh_command_queues_.reserve(this->num_hw_cqs());
    if (this->using_fast_dispatch()) {
        for (std::size_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
            mesh_command_queues_.push_back(std::make_unique<FDMeshCommandQueue>(
                this, cq_id, dispatch_thread_pool_, reader_thread_pool_, cq_shared_state));
        }
    } else {
        for (std::size_t cq_id = 0; cq_id < this->num_hw_cqs(); cq_id++) {
            mesh_command_queues_.push_back(std::make_unique<SDMeshCommandQueue>(this, cq_id));
        }
    }
    Inspector::mesh_device_initialized(this);
    return true;
}

void MeshDevice::init_command_queue_host() {
    TT_THROW("init_command_queue_host() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_host();
}
void MeshDevice::init_command_queue_device() {
    TT_THROW("init_command_queue_device() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_device();
}
bool MeshDevice::compile_fabric() {
    TT_THROW("compile_fabric() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->compile_fabric();
}
void MeshDevice::configure_fabric() {
    TT_THROW("configure_fabric() is not supported on MeshDevice - use individual devices instead");
    reference_device()->configure_fabric();
}
void MeshDevice::init_fabric() {
    TT_THROW("init_fabric_program() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_fabric();
}

program_cache::detail::ProgramCache& MeshDevice::get_program_cache() { return *program_cache_; }
HalProgrammableCoreType MeshDevice::get_programmable_core_type(CoreCoord virtual_core) const {
    return reference_device()->get_programmable_core_type(virtual_core);
}

HalMemType MeshDevice::get_mem_type_of_core(CoreCoord virtual_core) const {
    return reference_device()->get_mem_type_of_core(virtual_core);
}

// Methods for SubDevice Management
uint8_t MeshDevice::num_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_mcast_txns(sub_device_id);
}
uint8_t MeshDevice::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_unicast_txns(sub_device_id);
}
uint8_t MeshDevice::noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data, bool unicast_data) const {
    if (mcast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_data_start_index(sub_device_id);
    } else if (unicast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_unicast_data_start_index(
            sub_device_id);
    } else {
        return 0;
    }
}
SubDeviceManagerId MeshDevice::get_active_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->id();
}
SubDeviceManagerId MeshDevice::get_default_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->id();
}
CoreCoord MeshDevice::virtual_program_dispatch_core(uint8_t cq_id) const {
    return validate_and_get_reference_value(
        this->get_devices(), [cq_id](const auto* device) { return device->virtual_program_dispatch_core(cq_id); });
}
const std::vector<SubDeviceId>& MeshDevice::get_sub_device_ids() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_ids();
}
const std::vector<SubDeviceId>& MeshDevice::get_sub_device_stall_group() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_stall_group();
}
void MeshDevice::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_manager_tracker_->get_active_sub_device_manager()->set_sub_device_stall_group(sub_device_ids);
}
void MeshDevice::reset_sub_device_stall_group() {
    sub_device_manager_tracker_->get_active_sub_device_manager()->reset_sub_device_stall_group();
}

uint32_t MeshDevice::num_sub_devices() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices();
}

bool MeshDevice::is_mmio_capable() const {
    TT_THROW("is_mmio_capable() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->is_mmio_capable();
}

// Allocator methods
std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address() const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address();
}

std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address(sub_device_ids);
}

const std::unique_ptr<Allocator>& MeshDevice::allocator() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->allocator(SubDeviceId{0});
}
const std::unique_ptr<Allocator>& MeshDevice::allocator(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->allocator(sub_device_id);
}

std::shared_ptr<distributed::MeshDevice> MeshDevice::get_mesh_device() { return shared_from_this(); }

MeshCoordinate MeshDevice::local_offset() const {
    return view_->local_offset();
}

MeshShape MeshDevice::local_shape() const {
    return view_->local_shape();
}

bool MeshDevice::is_local_coordinate(const MeshCoordinate& coord) const {
    return view_->is_local_coordinate(coord);
}

}  // namespace tt::tt_metal::distributed
