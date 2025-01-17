// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_device.hpp>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

#include <logger.hpp>
#include <host_api.hpp>
#include <tt_metal.hpp>
#include <system_mesh.hpp>
#include <mesh_device_view.hpp>
#include "tt_metal/distributed/mesh_command_queue.hpp"
#include <device_impl.hpp>
#include <sub_device.hpp>
#include <sub_device_manager_tracker.hpp>
#include <sub_device_manager.hpp>
#include <sub_device_types.hpp>

#include <hal.hpp>

namespace tt::tt_metal::distributed {

namespace {
MeshDeviceID generate_unique_mesh_id() {
    static std::atomic<MeshDeviceID> next_id{0};
    return next_id++;
}
}

MeshDevice::ScopedDevices::ScopedDevices(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    const MeshDeviceConfig& config) {
    auto& system_mesh = SystemMesh::instance();
    auto physical_device_ids = system_mesh.request_available_devices(config);

    opened_devices_ = tt::tt_metal::detail::CreateDevices(
        physical_device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config);

    for (auto physical_device_id : physical_device_ids) {
        devices_.push_back(opened_devices_.at(physical_device_id));
    }
}

MeshDevice::ScopedDevices::~ScopedDevices() {
    if (not opened_devices_.empty()) {
        tt::tt_metal::detail::CloseDevices(opened_devices_);
        opened_devices_.clear();
        devices_.clear();
    }
}

const std::vector<IDevice*>& MeshDevice::ScopedDevices::get_devices() const { return devices_; }

uint32_t MeshDevice::build_key() const {
    TT_FATAL(tt::tt_metal::hal.is_coordinate_virtualization_enabled(), "MeshDevice::build_key() expects coordinate virtualization to be enabled");
    return reference_device()->build_key();
}

uint8_t MeshDevice::num_hw_cqs() const { return reference_device()->num_hw_cqs(); }

bool MeshDevice::is_initialized() const {
    const auto& devices = this->get_devices();
    return std::all_of(devices.begin(), devices.end(), [](const auto& device) { return device->is_initialized(); });
}
uint32_t MeshDevice::l1_size_per_core() const { return reference_device()->l1_size_per_core(); }

uint32_t MeshDevice::dram_size_per_channel() const { return reference_device()->dram_size_per_channel(); }

IDevice* MeshDevice::reference_device() const { return this->get_devices().at(0); }

MeshDevice::MeshDevice(std::shared_ptr<ScopedDevices> mesh_handle, const MeshShape& mesh_shape, MeshType type, std::weak_ptr<MeshDevice> parent_mesh) :
    scoped_devices_(std::move(mesh_handle)),
    mesh_shape_(mesh_shape),
    type_(type),
    mesh_id_(generate_unique_mesh_id()),
    parent_mesh_(std::move(parent_mesh)) {}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    auto mesh_device = std::make_shared<MeshDevice>(
        std::make_shared<ScopedDevices>(l1_small_size, trace_region_size, num_command_queues, dispatch_core_config, config),
        config.mesh_shape,
        config.mesh_type);

    mesh_device->initialize(num_command_queues, l1_small_size, trace_region_size, l1_bank_remap);
    return mesh_device;
}

std::shared_ptr<MeshDevice> MeshDevice::create_submesh(
    const MeshShape& submesh_shape, const MeshOffset& offset, MeshType type) {
    if (submesh_shape.num_rows <= 0 || submesh_shape.num_cols <= 0) {
        TT_THROW(
            "Invalid submesh shape: ({}, {}). Both dimensions must be positive.",
            submesh_shape.num_rows,
            submesh_shape.num_cols);
    }

    if (offset.row < 0 || offset.col < 0) {
        TT_THROW("Invalid offset: ({}, {}). Offset must be non-negative.", offset.row, offset.col);
    }

    if (offset.row + submesh_shape.num_rows > mesh_shape_.num_rows ||
        offset.col + submesh_shape.num_cols > mesh_shape_.num_cols) {
        TT_THROW(
            "Submesh ({}x{}) with offset ({}, {}) does not fit within parent mesh ({}x{}).",
            submesh_shape.num_rows,
            submesh_shape.num_cols,
            offset.row,
            offset.col,
            mesh_shape_.num_rows,
            mesh_shape_.num_cols);
    }

    auto submesh = std::make_shared<MeshDevice>(scoped_devices_, submesh_shape, type, shared_from_this());
    auto start_coordinate = Coordinate{offset.row, offset.col};
    auto end_coordinate = Coordinate{offset.row + submesh_shape.num_rows - 1, offset.col + submesh_shape.num_cols - 1};

    auto submesh_devices = view_->get_devices(start_coordinate, end_coordinate);
    submesh->view_ = std::make_unique<MeshDeviceView>(submesh_devices, submesh_shape);
    SystemMesh::instance().register_mesh_device(submesh, submesh_devices);
    submeshes_.push_back(submesh);
    log_trace(
        LogMetal,
        "Instantiating submesh {}: {}x{} with offset: {} {}",
        submesh->id(),
        submesh_shape.num_rows,
        submesh_shape.num_cols,
        offset.row,
        offset.col);
    log_trace(LogMetal, "Submesh {} instantiated with {} devices", submesh->id(), submesh->get_devices().size());

    return submesh;
}

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::create_submeshes(const MeshShape& submesh_shape, MeshType type) {
    std::vector<std::shared_ptr<MeshDevice>> submeshes;
    for (int row = 0; row < this->num_rows(); row += submesh_shape.num_rows) {
        for (int col = 0; col < this->num_cols(); col += submesh_shape.num_cols) {
            auto submesh = this->create_submesh(submesh_shape, MeshOffset{row, col}, type);
            submeshes.push_back(submesh);
        }
    }
    return submeshes;
}

MeshDevice::~MeshDevice() {}

IDevice* MeshDevice::get_device_index(size_t device_index) const {
    TT_FATAL(device_index >= 0 and device_index < num_devices(), "Invalid device index");
    const auto& devices = scoped_devices_->get_devices();
    return devices.at(device_index);
}

IDevice* MeshDevice::get_device(chip_id_t physical_device_id) const {
    for (auto device : this->get_devices()) {
        if (device->id() == physical_device_id) {
            return device;
        }
    }
    TT_THROW("Physical Device ID: {} not found in assigned devices", physical_device_id);
}

std::vector<IDevice*> MeshDevice::get_devices(const std::optional<MeshType>& requested_type) const {
    return view_->get_devices(requested_type.value_or(type_));
}

// TODO: Remove this function once we have a proper view interface
IDevice* MeshDevice::get_device(size_t row_idx, size_t col_idx) const {
    return this->get_device_index(row_idx * num_cols() + col_idx);
}

MeshCommandQueue& MeshDevice::mesh_command_queue() {
    TT_FATAL(this->using_fast_dispatch(), "Can only acess the MeshCommandQueue when using Fast Dispatch.");
    return *(mesh_command_queue_);
}

const DeviceIds MeshDevice::get_device_ids() const {
    DeviceIds device_ids;
    for (auto device : this->get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

size_t MeshDevice::num_devices() const { return view_->num_devices(); }

CoreCoord MeshDevice::compute_with_storage_grid_size() const {
    return this->reference_device()->compute_with_storage_grid_size();
}


tt::ARCH MeshDevice::arch() const { return this->reference_device()->arch(); }

size_t MeshDevice::num_rows() const { return mesh_shape_.num_rows; }

size_t MeshDevice::num_cols() const { return mesh_shape_.num_cols; }

MeshShape MeshDevice::shape() const { return mesh_shape_; }

void MeshDevice::reshape(const MeshShape& new_shape) {
    TT_FATAL(
        new_shape.num_rows * new_shape.num_cols == this->num_devices(),
        "New shape must have the same number of devices as current shape");

    std::unordered_map<chip_id_t, size_t> physical_device_id_to_linearized_index;
    for (size_t i = 0; i < this->num_devices(); i++) {
        physical_device_id_to_linearized_index[this->get_devices()[i]->id()] = i;
    }

    // From an MxN mesh, we can always reduce rank to a 1xM*N Line mesh.
    // However, going from a Line mesh to an MxN mesh is not always possible.
    if (new_shape.num_rows != 1 and new_shape.num_cols != 1) {
        auto new_physical_device_ids =
            SystemMesh::instance().request_available_devices(
                MeshDeviceConfig{
                    .mesh_shape=new_shape
                }
            );

        for (size_t i = 0; i < new_physical_device_ids.size(); i++) {
            if (physical_device_id_to_linearized_index.find(new_physical_device_ids[i]) == physical_device_id_to_linearized_index.end()) {
                TT_THROW(
                    "User has requested a reshape of the MeshDevice to shape: {}x{}, but it is not possible to form a "
                    "physically connected mesh of {}x{} grid with the opened devices from the original shape: {}x{}.",
                    new_shape.num_rows,
                    new_shape.num_cols,
                    new_shape.num_rows,
                    new_shape.num_cols,
                    this->num_rows(),
                    this->num_cols());
            }
        }
    }

    mesh_shape_ = new_shape;
    view_ = std::make_unique<MeshDeviceView>(scoped_devices_->get_devices(), mesh_shape_);
}

bool MeshDevice::close() {
    for (const auto& submesh : submeshes_) {
        submesh->close();
    }
    submeshes_.clear();
    if (scoped_devices_) {
        scoped_devices_.reset();
    }
    parent_mesh_.reset();
    view_.reset();
    sub_device_manager_tracker_.reset();
    return true;
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", this->num_rows(), this->num_cols(), this->num_devices());
}

const MeshDeviceView& MeshDevice::get_view() const {
    TT_FATAL(view_, "MeshDeviceView is not initialized");
    return *view_;
}

MeshDeviceID MeshDevice::id() const { return mesh_id_; }

bool MeshDevice::is_parent_mesh() const { return parent_mesh_.expired(); }

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::get_submeshes() const { return submeshes_; }

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

void MeshDevice::enable_async(bool enable) {
    for (auto device : this->get_devices()) {
        device->enable_async(enable);
    }
}

void MeshDevice::enable_program_cache() {
    for (auto device : this->get_devices()) {
        device->enable_program_cache();
    }
}

void MeshDevice::disable_and_clear_program_cache() {
    for (auto device : this->get_devices()) {
        device->disable_and_clear_program_cache();
    }
}

size_t MeshDevice::num_program_cache_entries() {
    size_t total_entries = 0;
    for (auto device : this->get_devices()) {
        total_entries += device->num_program_cache_entries();
    }
    return total_entries;
}

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

std::tuple<SubDeviceManagerId, SubDeviceId> MeshDevice::create_sub_device_manager_with_fabric(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
}
CoreCoord MeshDevice::dram_grid_size() const { return reference_device()->dram_grid_size(); }

bool MeshDevice::using_slow_dispatch() const {
    const auto& devices = this->get_devices();
    TT_FATAL(devices.size() > 0, "Expected at least one device in a Mesh to use slow dispatch.");
    bool first_device_dispatch = devices[0]->using_slow_dispatch();
    TT_FATAL(std::all_of(
        devices.begin(),
        devices.end(),
        [first_device_dispatch](const IDevice* device) {
            return device->using_slow_dispatch() == first_device_dispatch;
        }), "Expected all devices in a Mesh to use identical dispatch modes.");
    return first_device_dispatch;
}

bool MeshDevice::using_fast_dispatch() const {
    const auto& devices = this->get_devices();
    TT_FATAL(devices.size() > 0, "Expected at least one device in a Mesh to use fast dispatch.");
    bool first_device_dispatch = devices[0]->using_fast_dispatch();
    TT_FATAL(std::all_of(
        devices.begin(),
        devices.end(),
        [first_device_dispatch](const IDevice* device) {
            return device->using_fast_dispatch() == first_device_dispatch;
        }), "Expected all devices in a Mesh to use identical dispatch modes.");
    return first_device_dispatch;
}

// Device property methods that can be delegated to reference device
CoreCoord MeshDevice::grid_size() const { return reference_device()->grid_size(); }
CoreCoord MeshDevice::logical_grid_size() const { return reference_device()->logical_grid_size(); }
CoreType MeshDevice::core_type_from_virtual_core(const CoreCoord& virtual_coord) const { return reference_device()->core_type_from_virtual_core(virtual_coord); }
CoreCoord MeshDevice::virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const { return reference_device()->virtual_noc_coordinate(noc_index, coord); }
CoreCoord MeshDevice::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const { return reference_device()->virtual_noc0_coordinate(noc_index, coord); }
std::vector<CoreCoord> MeshDevice::worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const {
    return reference_device()->worker_cores_from_logical_cores(logical_cores);
}
std::vector<CoreCoord> MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment() { return reference_device()->get_optimal_dram_bank_to_logical_worker_assignment(); }
CoreCoord MeshDevice::virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const { return reference_device()->virtual_core_from_logical_core(logical_coord, core_type); }
CoreCoord MeshDevice::worker_core_from_logical_core(const CoreCoord& logical_core) const {
    return reference_device()->worker_core_from_logical_core(logical_core);
}
CoreCoord MeshDevice::logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const {
    return reference_device()->logical_core_from_ethernet_core(ethernet_core);
}

// These methods require some change / or assert out for now
std::vector<CoreCoord> MeshDevice::ethernet_cores_from_logical_cores(
    const std::vector<CoreCoord>& logical_cores) const {
    return reference_device()->ethernet_cores_from_logical_cores(logical_cores);
}
CoreCoord MeshDevice::ethernet_core_from_logical_core(const CoreCoord& logical_core) const {
    TT_THROW("ethernet_core_from_logical_core() is not supported on MeshDevice - use individual devices instead");
}
std::unordered_set<CoreCoord> MeshDevice::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    TT_THROW("get_active_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

std::unordered_set<CoreCoord> MeshDevice::get_inactive_ethernet_cores() const {
    TT_THROW("get_inactive_ethernet_cores() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDevice::is_inactive_ethernet_core(CoreCoord logical_core) const {
    TT_THROW("is_inactive_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::tuple<chip_id_t, CoreCoord> MeshDevice::get_connected_ethernet_core(CoreCoord eth_core) const {
    TT_THROW("get_connected_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

bool MeshDevice::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    TT_THROW("is_active_ethernet_core() is not supported on MeshDevice - use individual devices instead");
}

std::vector<CoreCoord> MeshDevice::get_ethernet_sockets(chip_id_t connected_chip_id) const {
    TT_THROW("get_ethernet_sockets() is not supported on MeshDevice - use individual devices instead");
}

// Core and worker management methods (These are OK)
CoreRangeSet MeshDevice::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}
uint32_t MeshDevice::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).num_cores(core_type);
}

// Bank and memory management methods
int MeshDevice::num_dram_channels() const { return reference_device()->num_dram_channels() * this->num_devices(); }
uint32_t MeshDevice::num_banks(const BufferType& buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::num_banks(*allocator, buffer_type);
}
uint32_t MeshDevice::num_banks(const BufferType& buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::num_banks(*allocator, buffer_type);
}
uint32_t MeshDevice::bank_size(const BufferType& buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_size(*allocator, buffer_type);
}
uint32_t MeshDevice::bank_size(const BufferType& buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_size(*allocator, buffer_type);
}
uint32_t MeshDevice::dram_channel_from_bank_id(uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::dram_channel_from_bank_id(*allocator, bank_id);
}
uint32_t MeshDevice::dram_channel_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::dram_channel_from_bank_id(*allocator, bank_id);
}
CoreCoord MeshDevice::logical_core_from_dram_channel(uint32_t dram_channel) const {
    return this->reference_device()->logical_core_from_dram_channel(dram_channel);
}
uint32_t MeshDevice::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    return this->reference_device()->dram_channel_from_logical_core(logical_core);
}
int32_t MeshDevice::bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_offset(*allocator, buffer_type, bank_id);
}
int32_t MeshDevice::bank_offset(BufferType buffer_type, uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_offset(*allocator, buffer_type, bank_id);
}
CoreCoord MeshDevice::logical_core_from_bank_id(uint32_t bank_id) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::logical_core_from_bank_id(*allocator, bank_id);
}

CoreCoord MeshDevice::logical_core_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::logical_core_from_bank_id(*allocator, bank_id);
}
const std::vector<uint32_t>& MeshDevice::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_ids_from_dram_channel(*allocator, dram_channel);
}
const std::vector<uint32_t>& MeshDevice::bank_ids_from_dram_channel(
    uint32_t dram_channel, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_ids_from_dram_channel(*allocator, dram_channel);
}
const std::vector<uint32_t>& MeshDevice::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord& logical_core) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::bank_ids_from_logical_core(*allocator, buffer_type, logical_core);
}
const std::vector<uint32_t>& MeshDevice::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord& logical_core, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::bank_ids_from_logical_core(*allocator, buffer_type, logical_core);
}

// Core management and network operations
const std::set<CoreCoord>& MeshDevice::ethernet_cores() const { return reference_device()->ethernet_cores(); }
const std::set<CoreCoord>& MeshDevice::storage_only_cores() const { return reference_device()->storage_only_cores(); }
uint32_t MeshDevice::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const { return reference_device()->get_noc_unicast_encoding(noc_index, core); }
uint32_t MeshDevice::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const { return reference_device()->get_noc_multicast_encoding(noc_index, cores); }

// Floating point and build environment
const JitBuildEnv& MeshDevice::build_env() const {
    TT_THROW("build_env() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_env();
}

// Build and firmware paths
const string MeshDevice::build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const {
    TT_THROW("build_firmware_target_path() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_firmware_target_path(programmable_core, processor_class, i);
}
const string MeshDevice::build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const {
    TT_THROW("build_kernel_target_path() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_kernel_target_path(programmable_core, processor_class, i, kernel_name);
}
const JitBuildState& MeshDevice::build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const {
    TT_THROW("build_firmware_state() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_firmware_state(programmable_core, processor_class, i);
}
const JitBuildState& MeshDevice::build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const {
    TT_THROW("build_kernel_state() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_kernel_state(programmable_core, processor_class, i);
}
const JitBuildStateSubset MeshDevice::build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const {
    TT_THROW("build_kernel_states() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->build_kernel_states(programmable_core, processor_class);
}

// System memory and command queue management
SystemMemoryManager& MeshDevice::sysmem_manager() {
    TT_THROW("sysmem_manager() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->sysmem_manager();
}
HWCommandQueue& MeshDevice::hw_command_queue(size_t cq_id) {
    TT_THROW("hw_command_queue() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->hw_command_queue(cq_id);
}
CommandQueue& MeshDevice::command_queue(size_t cq_id) {
    TT_THROW("command_queue() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->command_queue(cq_id);
}

// Trace management
void MeshDevice::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_THROW("begin_trace() is not supported on MeshDevice - use individual devices instead");
    reference_device()->begin_trace(cq_id, tid);
}
void MeshDevice::end_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_THROW("end_trace() is not supported on MeshDevice - use individual devices instead");
    reference_device()->end_trace(cq_id, tid);
}
void MeshDevice::replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    TT_THROW("replay_trace() is not supported on MeshDevice - use individual devices instead");
    reference_device()->replay_trace(cq_id, tid, blocking);
}
void MeshDevice::release_trace(const uint32_t tid) {
    TT_THROW("release_trace() is not supported on MeshDevice - use individual devices instead");
    reference_device()->release_trace(tid);
}
std::shared_ptr<TraceBuffer> MeshDevice::get_trace(uint32_t tid) {
    TT_THROW("get_trace() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_trace(tid);
}
uint32_t MeshDevice::get_trace_buffers_size() const {
    TT_THROW("get_trace_buffers_size() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_trace_buffers_size();
}
void MeshDevice::set_trace_buffers_size(uint32_t size) {
    TT_THROW("set_trace_buffers_size() is not supported on MeshDevice - use individual devices instead");
    reference_device()->set_trace_buffers_size(size);
}

// Dispatch and initialization
bool MeshDevice::initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap, bool minimal) {
    view_ = std::make_unique<MeshDeviceView>(scoped_devices_->get_devices(), mesh_shape_);
    SystemMesh::instance().register_mesh_device(shared_from_this(), this->get_devices());
    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        this,
        dynamic_cast<Device*>(reference_device())
            ->initialize_allocator(l1_small_size, trace_region_size, l1_bank_remap));
    if (this->using_fast_dispatch()) {
        mesh_command_queue_ = std::make_unique<MeshCommandQueue>(this, 0);
    }
    return true;
}

void MeshDevice::build_firmware() {
    TT_THROW("build_firmware() is not supported on MeshDevice - use individual devices instead");
    reference_device()->build_firmware();
}
void MeshDevice::reset_cores() {
    TT_THROW("reset_cores() is not supported on MeshDevice - use individual devices instead");
    reference_device()->reset_cores();
}
void MeshDevice::initialize_and_launch_firmware() {
    TT_THROW("initialize_and_launch_firmware() is not supported on MeshDevice - use individual devices instead");
    reference_device()->initialize_and_launch_firmware();
}
void MeshDevice::init_command_queue_host() {
    TT_THROW("init_command_queue_host() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_host();
}
void MeshDevice::init_command_queue_device() {
    TT_THROW("init_command_queue_device() is not supported on MeshDevice - use individual devices instead");
    reference_device()->init_command_queue_device();
}
void MeshDevice::initialize_synchronous_sw_cmd_queue() {
    TT_THROW("initialize_synchronous_sw_cmd_queue() is not supported on MeshDevice - use individual devices instead");
    reference_device()->initialize_synchronous_sw_cmd_queue();
}
void MeshDevice::update_dispatch_cores_for_multi_cq_eth_dispatch() {
    TT_THROW("update_dispatch_cores_for_multi_cq_eth_dispatch() is not supported on MeshDevice - use individual devices instead");
    reference_device()->update_dispatch_cores_for_multi_cq_eth_dispatch();
}
void MeshDevice::synchronize() {
    TT_THROW("synchronize() is not supported on MeshDevice - use individual devices instead");
    reference_device()->synchronize();
}
WorkExecutorMode MeshDevice::get_worker_mode() {
    TT_THROW("get_worker_mode() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_worker_mode();
}
void MeshDevice::set_worker_queue_mode(const WorkerQueueMode& mode) {
    TT_THROW("set_worker_queue_mode() is not supported on MeshDevice - use individual devices instead");
    reference_device()->set_worker_queue_mode(mode);
}
WorkerQueueMode MeshDevice::get_worker_queue_mode() {
    TT_THROW("get_worker_queue_mode() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_worker_queue_mode();
}
bool MeshDevice::is_worker_queue_empty() const {
    TT_THROW("is_worker_queue_empty() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->is_worker_queue_empty();
}
bool MeshDevice::can_use_passthrough_scheduling() const {
    TT_THROW("can_use_passthrough_scheduling() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->can_use_passthrough_scheduling();
}
void MeshDevice::push_work(std::function<void()> work, bool blocking) {
    TT_THROW("push_work() is not supported on MeshDevice - use individual devices instead");
    reference_device()->push_work(std::move(work), blocking);
}
program_cache::detail::ProgramCache& MeshDevice::get_program_cache() { return reference_device()->get_program_cache(); }
HalProgrammableCoreType MeshDevice::get_programmable_core_type(CoreCoord virtual_core) const { return reference_device()->get_programmable_core_type(virtual_core); }
std::vector<std::pair<transfer_info_cores, uint32_t>> MeshDevice::extract_dst_noc_multicast_info(const std::vector<CoreRange>& ranges, const CoreType core_type) {
    TT_THROW("extract_dst_noc_multicast_info() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->extract_dst_noc_multicast_info(ranges, core_type);
}
bool MeshDevice::dispatch_s_enabled() const {
    bool reference_dispatch_s_enabled = reference_device()->dispatch_s_enabled();
    TT_FATAL(
        std::all_of(
            scoped_devices_->get_devices().begin(),
            scoped_devices_->get_devices().end(),
            [&](const auto& device) { return device->dispatch_s_enabled() == reference_dispatch_s_enabled; }),
        "All devices must have the same dispatch_s_enabled value");
    return reference_dispatch_s_enabled;
}
bool MeshDevice::distributed_dispatcher() const {
    TT_THROW("distributed_dispatcher() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->distributed_dispatcher();
}
NOC MeshDevice::dispatch_go_signal_noc() const {
    NOC reference_dispatch_go_signal_noc = reference_device()->dispatch_go_signal_noc();
    TT_FATAL(
        std::all_of(
            scoped_devices_->get_devices().begin(),
            scoped_devices_->get_devices().end(),
            [&](const auto& device) { return device->dispatch_go_signal_noc() == reference_dispatch_go_signal_noc; }),
        "All devices must have the same dispatch_go_signal_noc value");
    return reference_dispatch_go_signal_noc;
}
size_t MeshDevice::get_device_kernel_defines_hash() {
    TT_THROW("get_device_kernel_defines_hash() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_device_kernel_defines_hash();
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
    return reference_device()->virtual_program_dispatch_core(cq_id);
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
std::optional<SubDeviceId> MeshDevice::get_fabric_sub_device_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->fabric_sub_device_id();
}
uint32_t MeshDevice::get_completion_queue_reader_core() const {
    TT_THROW("get_completion_queue_reader_core() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_completion_queue_reader_core();
}
bool MeshDevice::is_mmio_capable() const {
    TT_THROW("is_mmio_capable() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->is_mmio_capable();
}
std::vector<std::vector<chip_id_t>> MeshDevice::get_tunnels_from_mmio() const {
    TT_THROW("get_tunnels_from_mmio() is not supported on MeshDevice - use individual devices instead");
    return reference_device()->get_tunnels_from_mmio();
}

// Allocator methods
// Memory statistics and buffer management
uint32_t MeshDevice::get_allocator_alignment() const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator->config.alignment;
}
uint32_t MeshDevice::get_allocator_alignment(SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator->config.alignment;
}
std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address() const { return reference_device()->lowest_occupied_compute_l1_address(); }
std::optional<DeviceAddr> MeshDevice::lowest_occupied_compute_l1_address(
    tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    // Sub-device banks are currently all compute banks
    // Since banks are lockstep in an allocator it doesn't matter which core is used
    uint32_t sub_device_bank_id = 0;
    DeviceAddr lowest_addr = std::numeric_limits<DeviceAddr>::max();
    for (const auto& sub_device_id : sub_device_ids) {
        const auto& allocator =
            sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device_allocator(sub_device_id);
        if (allocator) {
            auto found_addr = allocator::lowest_occupied_l1_address(*allocator, sub_device_bank_id);
            if (found_addr.has_value()) {
                lowest_addr = std::min(lowest_addr, *found_addr);
            }
        }
    }
    // sub-device allocators sit below global allocator. If an address is found for a sub-device, no need to check the
    // global allocator
    if (lowest_addr != std::numeric_limits<DeviceAddr>::max()) {
        return lowest_addr;
    } else {
        const auto& allocator = this->get_initialized_allocator();
        auto reference_device = this->reference_device();
        auto logical_compute_cores = tt::get_logical_compute_cores(
            reference_device->id(),
            reference_device->num_hw_cqs(),
            dispatch_core_manager::instance().get_dispatch_core_config(reference_device->id()));
        TT_ASSERT(logical_compute_cores.size() > 0, "No logical compute cores found");
        // Global bank id needs to look up a bank from the compute grid (not the storage grid)
        auto global_bank_id = this->bank_ids_from_logical_core(BufferType::L1, logical_compute_cores[0])[0];
        return allocator::lowest_occupied_l1_address(*allocator, global_bank_id);
    }
}

size_t MeshDevice::get_l1_small_size() const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator->config.l1_small_size;
}
size_t MeshDevice::get_l1_small_size(SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator->config.l1_small_size;
}
const std::unordered_set<Buffer*>& MeshDevice::get_allocated_buffers() const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_allocated_buffers(*allocator);
}
const std::unordered_set<Buffer*>& MeshDevice::get_allocated_buffers(SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_allocated_buffers(*allocator);
}
allocator::Statistics MeshDevice::get_memory_allocation_statistics(const BufferType& buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_statistics(*allocator, buffer_type);
}

allocator::Statistics MeshDevice::get_memory_allocation_statistics(
    const BufferType& buffer_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_statistics(*allocator, buffer_type);
}
const std::unique_ptr<Allocator>& MeshDevice::get_initialized_allocator() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->get_initialized_allocator(SubDeviceId{0});
}
const std::unique_ptr<Allocator>& MeshDevice::get_initialized_allocator(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_initialized_allocator(sub_device_id);
}
DeviceAddr MeshDevice::get_base_allocator_addr(const HalMemType& mem_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_unreserved_base_address(*allocator, mem_type);
}
DeviceAddr MeshDevice::get_base_allocator_addr(const HalMemType& mem_type, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    return allocator::get_unreserved_base_address(*allocator, mem_type);
}
// Buffer and memory management operations
void MeshDevice::deallocate_buffers() {
    const auto& allocator = this->get_initialized_allocator();
    allocator::deallocate_buffers(*allocator);
}
void MeshDevice::deallocate_buffers(SubDeviceId sub_device_id) {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    allocator::deallocate_buffers(*allocator);
}
void MeshDevice::dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out) const {
    const auto& allocator = this->get_initialized_allocator();
    allocator::dump_memory_blocks(*allocator, buffer_type, out);
}
void MeshDevice::dump_memory_blocks(
    const BufferType& buffer_type, std::ofstream& out, SubDeviceId sub_device_id) const {
    const auto& allocator = this->get_initialized_allocator(sub_device_id);
    allocator::dump_memory_blocks(*allocator, buffer_type, out);
}
MemoryBlockTable MeshDevice::get_memory_block_table(const BufferType& buffer_type) const {
    const auto& allocator = this->get_initialized_allocator();
    return allocator::get_memory_block_table(*allocator, buffer_type);
}

MeshSubDeviceManagerId MeshDevice::mesh_create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    MeshSubDeviceManagerId mesh_sub_device_manager_id(*this);
    const auto& devices = scoped_devices_->get_devices();
    for (uint32_t i = 0; i < devices.size(); i++) {
        auto* device = devices[i];
        auto& sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work([device, sub_devices, local_l1_size, &sub_device_manager_id]() {
            sub_device_manager_id = device->create_sub_device_manager(sub_devices, local_l1_size);
        });
    }
    for (auto* device : devices) {
        device->synchronize();
    }
    return mesh_sub_device_manager_id;
}

std::tuple<MeshSubDeviceManagerId, SubDeviceId> MeshDevice::mesh_create_sub_device_manager_with_fabric(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    MeshSubDeviceManagerId mesh_sub_device_manager_id(*this);
    SubDeviceId fabric_sub_device_id;
    const auto& devices = scoped_devices_->get_devices();
    for (uint32_t i = 0; i < devices.size(); i++) {
        auto* device = devices[i];
        auto& sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        // All fabric sub-device ids will be the same, since all managers are created with the same sub_devices input
        device->push_work([device, sub_devices, local_l1_size, &sub_device_manager_id, &fabric_sub_device_id]() {
            std::tie(sub_device_manager_id, fabric_sub_device_id) = device->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
        });
    }
    for (auto* device : devices){
        device->synchronize();
    }
    return {mesh_sub_device_manager_id, fabric_sub_device_id};
}

void MeshDevice::mesh_load_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id) {
    const auto& devices = scoped_devices_->get_devices();
    for (uint32_t i = 0; i < devices.size(); i++) {
        auto* device = devices[i];
        auto sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work(
            [device, sub_device_manager_id]() { device->load_sub_device_manager(sub_device_manager_id); });
    }
}
void MeshDevice::mesh_clear_loaded_sub_device_manager() {
    for (auto* device : scoped_devices_->get_devices()) {
        device->push_work([device]() { device->clear_loaded_sub_device_manager(); });
    }
}
void MeshDevice::mesh_remove_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id) {
    const auto& devices = scoped_devices_->get_devices();
    for (uint32_t i = 0; i < devices.size(); i++) {
        auto* device = devices[i];
        auto sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work(
            [device, sub_device_manager_id]() { device->remove_sub_device_manager(sub_device_manager_id); });
    }
}

void MeshDevice::mesh_set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    for (auto* device : scoped_devices_->get_devices()) {
        device->push_work([device, sub_device_ids=std::vector<SubDeviceId>(sub_device_ids.begin(), sub_device_ids.end())]() { device->set_sub_device_stall_group(sub_device_ids); });
    }
}

void MeshDevice::mesh_reset_sub_device_stall_group() {
    for (auto* device : scoped_devices_->get_devices()) {
        device->push_work([device]() { device->reset_sub_device_stall_group(); });
    }
}

MeshSubDeviceManagerId::MeshSubDeviceManagerId(const MeshDevice& mesh_device) {
    this->sub_device_manager_ids.resize(mesh_device.num_devices());
}


}  // namespace tt::tt_metal::distributed
