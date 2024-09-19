// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/mesh_device.hpp"

#include <memory>
#include <unordered_map>

#include "device/tt_cluster_descriptor_types.h"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"

namespace tt::tt_metal {

using LogicalCoordinate = Coordinate;
using PhysicalCoordinate = eth_coord_t;

static std::string get_config_path(const std::string& filename) {
    std::string root_path = getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : "./";
    return root_path + "/tt_metal/impl/device/mesh_configurations/" + filename;
}

static std::map<LogicalCoordinate, PhysicalCoordinate> load_translation_map(const std::string& filename, const std::string& key) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parsing error in file " + filename + ": " + e.what());
    }

    if (!j.contains(key)) {
        throw std::runtime_error("Key '" + key + "' not found in JSON file: " + filename);
    }

    std::map<LogicalCoordinate, PhysicalCoordinate> result;
    for (const auto& mapping : j[key]) {
        if (mapping.size() != 2 || mapping[0].size() != 2 || mapping[1].size() != 4) {
            throw std::runtime_error("Invalid coordinate format in JSON file: " + filename);
        }
        result.emplace(LogicalCoordinate{mapping[0][0], mapping[0][1]}, PhysicalCoordinate{mapping[1][1], mapping[1][0], mapping[1][2], mapping[1][3]});
    }

    return result;
}

MeshShape SystemMesh::get_system_mesh_shape(std::size_t system_num_devices) {
    const std::unordered_map<std::size_t, MeshShape> system_mesh_to_shape = {
        {1, MeshShape{1, 1}},   // single-device
        {2, MeshShape{1, 2}},   // N300
        {8, MeshShape{2, 4}},   // T3000; as ring to match existing tests
        {32, MeshShape{8, 4}},  // TG
        {64, MeshShape{8, 8}},  // TGG
    };
    TT_FATAL(system_mesh_to_shape.contains(system_num_devices), "Unsupported number of devices: {}", system_num_devices);
    auto shape = system_mesh_to_shape.at(system_num_devices);
    log_debug(LogMetal, "Logical SystemMesh Shape: {}x{}", shape.first, shape.second);
    return shape;
}

std::map<LogicalCoordinate, PhysicalCoordinate> SystemMesh::get_system_mesh_translation_map(std::size_t system_num_devices) {
    const std::unordered_map<std::size_t, std::string> system_mesh_translation_map = {
        {1, "device.json"},
        {2, "N300.json"},
        {8, "T3000.json"},
        {32, "TG.json"},
        {64, "TGG.json"},
    };
    TT_FATAL(system_mesh_translation_map.contains(system_num_devices), "Unsupported number of devices: {}", system_num_devices);
    auto translation_config_file = get_config_path(system_mesh_translation_map.at(system_num_devices));
    return load_translation_map(translation_config_file, "logical_to_physical_coordinates");
}

bool SystemMesh::is_system_mesh_initialized() const {
    return this->physical_coordinate_to_device_id.size() > 0;
}

SystemMesh& SystemMesh::instance() {
    static SystemMesh instance;
    if (!instance.is_system_mesh_initialized()) {
        instance.initialize();
    }
    return instance;
}
void SystemMesh::initialize() {
    this->physical_device_id_to_coordinate = tt::Cluster::instance().get_user_chip_ethernet_coordinates();
    for (const auto& [chip_id, physical_coordinate] : this->physical_device_id_to_coordinate) {
        this->physical_coordinate_to_device_id.emplace(physical_coordinate, chip_id);
    }

    // Initialize the system mesh shape and translation map
    auto num_devices = physical_coordinate_to_device_id.size();
    this->logical_mesh_shape = SystemMesh::get_system_mesh_shape(num_devices);
    this->logical_to_physical_coordinates = SystemMesh::get_system_mesh_translation_map(num_devices);
}

const MeshShape& SystemMesh::get_shape() const { return this->logical_mesh_shape; }
std::size_t SystemMesh::get_num_devices() const {
    auto [num_rows, num_cols] = this->get_shape();
    return num_rows * num_cols;
}

std::vector<chip_id_t> SystemMesh::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    std::vector<chip_id_t> physical_device_ids;
    auto [system_mesh_rows, system_mesh_cols] = this->get_shape();
    auto [requested_rows, requested_cols] = config.mesh_shape;
    auto [row_offset, col_offset] = config.offset;

    for (int row = 0; row < requested_rows; row++) {
        for (int col = 0; col < requested_cols; col++) {
            auto logical_device_id = (row + row_offset) * system_mesh_cols + (col + col_offset);
            auto logical_coordinate = Coordinate{logical_device_id / system_mesh_cols, logical_device_id % system_mesh_cols};
            auto physical_coordinate = this->logical_to_physical_coordinates.at(logical_coordinate);
            auto physical_device_id = this->physical_coordinate_to_device_id.at(physical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(LogMetal, "Logical device ID: {}, Logical coordinate: {}, Physical coordinate: {}, Physical device ID: {}",
                     logical_device_id, logical_coordinate, physical_coordinate, physical_device_id);
        }
    }
    return physical_device_ids;
}

std::vector<Device*> SystemMesh::map_mesh_device(
    std::shared_ptr<MeshDevice> mesh_device,
    size_t num_command_queues,
    size_t l1_small_size,
    size_t trace_region_size,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset,
    const std::vector<chip_id_t>& user_provided_physical_device_ids) {

    auto [requested_num_rows, requested_num_cols] = mesh_device->shape();
    auto [max_num_rows, max_num_cols] = this->logical_mesh_shape;
    auto [row_offset, col_offset] = offset;

    log_debug(LogMetal, "Mapping MeshDevice ({}x{}) with offset: {}, {}", requested_num_rows, requested_num_cols, row_offset, col_offset);
    TT_FATAL(requested_num_rows <= max_num_rows, "Requested too many rows: {} > {}", requested_num_rows, max_num_rows);
    TT_FATAL(requested_num_rows*requested_num_cols <= max_num_rows*max_num_cols, "Requested submesh is too big: {}x{}", requested_num_rows, requested_num_cols);

    this->assigned_mesh_device_devices.insert({mesh_device->get_mesh_id(), mesh_device});

    auto physical_device_ids = user_provided_physical_device_ids.empty() ?
        this->get_mapped_physical_device_ids(MeshDeviceConfig{mesh_device->shape(), offset}) :
        user_provided_physical_device_ids;

    this->opened_devices[mesh_device->get_mesh_id()] = tt::tt_metal::detail::CreateDevices(
        physical_device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_type);

    std::vector<Device*> mapped_devices;
    for (auto physical_device_id : physical_device_ids) {
        auto mapped_device = this->opened_devices[mesh_device->get_mesh_id()].at(physical_device_id);
        mapped_devices.push_back(mapped_device);
        this->assigned_devices[mesh_device->get_mesh_id()].push_back(physical_device_id);
        this->assigned_physical_id_to_device.insert({physical_device_id, mapped_device});
    }
    return mapped_devices;
}

void SystemMesh::unmap_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device) {
    auto mesh_id = mesh_device->get_mesh_id();

    // Clean up all state related to this virtual mesh
    this->assigned_mesh_device_devices.erase(mesh_id);

    // Remove the devices from assigned_physical_id_to_device
    for (auto physical_id : this->assigned_devices.at(mesh_id)) {
        this->assigned_physical_id_to_device.erase(physical_id);
    }
    this->assigned_devices.erase(mesh_id);

    // Close the devices
    tt::tt_metal::detail::CloseDevices(this->opened_devices.at(mesh_id));
    this->opened_devices.erase(mesh_id);
}

static MeshDeviceID generate_unique_mesh_id() {
    static std::atomic<MeshDeviceID> next_id{0};
    return next_id++;
}

MeshDevice::MeshDevice(const MeshShape& mesh_device_shape) : mesh_device_shape(mesh_device_shape), mesh_id(generate_unique_mesh_id()) {}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshShape& mesh_device_shape,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset,
    const std::vector<chip_id_t>& user_provided_physical_device_ids)
{
    auto mesh_device = std::make_shared<MeshDevice>(mesh_device_shape);
    mesh_device->initialize(l1_small_size, trace_region_size, num_command_queues, dispatch_core_type, offset, user_provided_physical_device_ids);

    return mesh_device;
}

void MeshDevice::initialize(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    DispatchCoreType dispatch_core_type,
    const std::pair<size_t, size_t>& offset,
    const std::vector<chip_id_t>& physical_device_ids)
{
    auto [num_rows, num_cols] = this->shape();
    auto num_requested_devices = num_rows * num_cols;
    auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();
    TT_FATAL(
        num_requested_devices <= num_available_devices,
        "User has requested more devices than available: {} requested, {} available",
        num_requested_devices, num_available_devices);

    auto& instance = SystemMesh::instance();
    this->devices = instance.map_mesh_device(
        shared_from_this(), num_command_queues, l1_small_size, trace_region_size, dispatch_core_type, offset, physical_device_ids);
    this->primary_view = std::make_unique<tt::tt_metal::MeshDeviceView>(*this);

    for (int device_index = 0; device_index < this->devices.size(); device_index++) {
        this->physical_id_to_device_index.insert({this->devices[device_index]->id(), device_index});
    }
}

MeshDevice::~MeshDevice() {
    if (not this->devices.empty()) {
        this->close_devices();
    }
}

Device* MeshDevice::get_device_index(int logical_device_id) const {
    TT_FATAL(logical_device_id >= 0 and logical_device_id < num_devices(), "Invalid device index");
    return this->devices.at(logical_device_id);
}

Device* MeshDevice::get_device(int physical_device_id) const {
    return this->devices.at(this->physical_id_to_device_index.at(physical_device_id));
}

std::vector<Device*> MeshDevice::get_devices() const { return this->devices; }

Device* MeshDevice::get_device(int row_idx, int col_idx) const {
    return this->get_device_index(row_idx * num_cols() + col_idx);
}

std::vector<Device*> MeshDevice::get_devices_on_row(int row_idx) const {
    return this->primary_view->get_devices_on_row(row_idx);
}

std::vector<Device*> MeshDevice::get_devices_on_column(int col_idx) const {
    return this->primary_view->get_devices_on_column(col_idx);
}

const DeviceIds MeshDevice::get_device_ids() const {
    DeviceIds device_ids;
    for (auto device : this->get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

int MeshDevice::num_devices() const { return num_rows() * num_cols(); }

CoreCoord MeshDevice::compute_with_storage_grid_size() const { return get_device_index(0)->compute_with_storage_grid_size(); }

CoreCoord MeshDevice::dram_grid_size() const { return get_device_index(0)->dram_grid_size(); }

tt::ARCH MeshDevice::arch() const { return get_device_index(0)->arch(); }

int MeshDevice::num_rows() const { return this->mesh_device_shape.first; }

int MeshDevice::num_cols() const { return this->mesh_device_shape.second; }

MeshShape MeshDevice::shape() const { return this->mesh_device_shape; }

void MeshDevice::close_devices() {
    SystemMesh::instance().unmap_mesh_device(shared_from_this());
    this->devices.clear();
    this->physical_id_to_device_index.clear();
    this->primary_view.reset();
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", this->num_rows(), this->num_cols(), this->num_devices());
}

std::shared_ptr<const MeshDeviceView> MeshDevice::get_view() const { return this->primary_view; }

std::shared_ptr<MeshDeviceView> MeshDevice::get_view() { return this->primary_view; }

MeshDeviceID MeshDevice::get_mesh_id() const { return this->mesh_id; }

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

bool validate_worker_modes(const std::vector<Device*>& workers) {
    bool worker_modes_match = true;
    auto first_worker_mode = workers.at(0)->get_worker_mode();
    for (auto worker : workers) {
        worker_modes_match &= (worker->get_worker_mode() == first_worker_mode);
    }
    return worker_modes_match;
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    auto& instance = SystemMesh::instance();
    auto num_devices = instance.get_num_devices();
    TT_FATAL(num_devices == 8, "T3000 ring topology only works with 8 devices");

    auto physical_device_ids = instance.get_mapped_physical_device_ids(MeshDeviceConfig{instance.get_shape(), MeshOffset{0, 0}});
    return physical_device_ids;
}

}  // namespace tt::tt_metal
