// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_device.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_shape.hpp"

namespace tt::tt_metal::distributed {
namespace {

using LogicalCoordinate = Coordinate;
using PhysicalCoordinate = eth_coord_t;

std::string get_config_path(const std::string& filename) {
    std::string root_path = getenv("TT_METAL_HOME") ? getenv("TT_METAL_HOME") : "./";
    return root_path + "/tt_metal/distributed/mesh_configurations/" + filename;
}

std::unordered_map<LogicalCoordinate, PhysicalCoordinate> load_translation_map(
    const std::string& filename, const std::string& key) {
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

    std::unordered_map<LogicalCoordinate, PhysicalCoordinate> result;
    for (const auto& mapping : j[key]) {
        if (mapping.size() != 2 || mapping[0].size() != 2 || mapping[1].size() != 5) {
            throw std::runtime_error("Invalid coordinate format in JSON file: " + filename);
        }
        result.emplace(
            LogicalCoordinate{mapping[0][0], mapping[0][1]},
            PhysicalCoordinate{
                mapping[1][0],  // cluster_id
                mapping[1][2],  // x
                mapping[1][1],  // y
                mapping[1][3],  // rack
                mapping[1][4]   // shelf
            });
    }

    return result;
}

MeshDeviceID generate_unique_mesh_id() {
    static std::atomic<MeshDeviceID> next_id{0};
    return next_id++;
}

}  // namespace

class SystemMesh::Impl {
private:
    using LogicalCoordinate = Coordinate;
    using PhysicalCoordinate = eth_coord_t;

    std::unordered_map<MeshDeviceID, std::vector<chip_id_t>> assigned_devices_;
    std::unordered_map<MeshDeviceID, std::weak_ptr<MeshDevice>> assigned_mesh_device_devices_;

    MeshShape logical_mesh_shape_;
    std::unordered_map<LogicalCoordinate, PhysicalCoordinate> logical_to_physical_coordinates_;
    std::unordered_map<LogicalCoordinate, chip_id_t> logical_to_device_id_;
    std::unordered_map<PhysicalCoordinate, chip_id_t> physical_coordinate_to_device_id_;
    std::unordered_map<chip_id_t, PhysicalCoordinate> physical_device_id_to_coordinate_;

public:
    Impl() = default;
    ~Impl() = default;

    bool is_system_mesh_initialized() const;
    void initialize();
    const MeshShape& get_shape() const;
    size_t get_num_devices() const;
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig& config) const;
    std::vector<chip_id_t> request_available_devices(const MeshDeviceConfig& config) const;
    Device* get_device(const chip_id_t physical_device_id) const;
    void register_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<Device*>& devices);

    static MeshShape get_system_mesh_shape(size_t system_num_devices);
    static std::unordered_map<LogicalCoordinate, PhysicalCoordinate> get_system_mesh_translation_map(
        size_t system_num_devices);

    chip_id_t get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const;
};

// Implementation of private static methods
MeshShape SystemMesh::Impl::get_system_mesh_shape(size_t system_num_devices) {
    const std::unordered_map<size_t, MeshShape> system_mesh_to_shape = {
        {1, MeshShape{1, 1}},   // single-device
        {2, MeshShape{1, 2}},   // N300
        {8, MeshShape{2, 4}},   // T3000; as ring to match existing tests
        {32, MeshShape{8, 4}},  // TG, QG
        {64, MeshShape{8, 8}},  // TGG
    };
    TT_FATAL(
        system_mesh_to_shape.contains(system_num_devices), "Unsupported number of devices: {}", system_num_devices);
    auto shape = system_mesh_to_shape.at(system_num_devices);
    log_debug(LogMetal, "Logical SystemMesh Shape: {}x{}", shape.num_rows, shape.num_cols);
    return shape;
}

std::unordered_map<LogicalCoordinate, PhysicalCoordinate> SystemMesh::Impl::get_system_mesh_translation_map(
    size_t system_num_devices) {
    // TG has 32 non-mmio user devices and 4 mmio devices not exposed to the user
    // QG has 32 mmio user devices
    // Once TG is fully deprecated, can remove TG code path
    std::string galaxy_mesh_descriptor = "TG.json";
    if (tt::Cluster::instance().number_of_pci_devices() == system_num_devices) {
        galaxy_mesh_descriptor = "QG.json";
    }
    const std::unordered_map<size_t, std::string> system_mesh_translation_map = {
        {1, "device.json"},
        {2, "N300.json"},
        {8, "T3000.json"},
        {32, galaxy_mesh_descriptor},
        {64, "TGG.json"},
    };
    TT_FATAL(
        system_mesh_translation_map.contains(system_num_devices),
        "Unsupported number of devices: {}",
        system_num_devices);
    auto translation_config_file = get_config_path(system_mesh_translation_map.at(system_num_devices));
    return load_translation_map(translation_config_file, "logical_to_physical_coordinates");
}

// Implementation of public methods
bool SystemMesh::Impl::is_system_mesh_initialized() const { return !physical_coordinate_to_device_id_.empty(); }

void SystemMesh::Impl::initialize() {
    physical_device_id_to_coordinate_ = tt::Cluster::instance().get_user_chip_ethernet_coordinates();
    if (physical_device_id_to_coordinate_.empty()) {
        // Only WH has ethernet coordinates. Fabric will assign chip ids for BH
        auto arch = tt::Cluster::instance().arch();
        TT_FATAL(
            arch == ARCH::GRAYSKULL or arch == ARCH::BLACKHOLE,
            "Expected Wormhole chips to have ethernet coordinates assigned by cluster descriptor");
        const int num_detected_devices = tt::Cluster::instance().number_of_devices();
        for (auto chip_id = 0; chip_id < num_detected_devices; chip_id++) {
            PhysicalCoordinate coord{0, chip_id, 0, 0, 0};
            physical_device_id_to_coordinate_.emplace(chip_id, coord);
            physical_coordinate_to_device_id_.emplace(coord, chip_id);
        }
    } else {
        for (const auto& [chip_id, physical_coordinate] : physical_device_id_to_coordinate_) {
            physical_coordinate_to_device_id_.emplace(physical_coordinate, chip_id);
        }
    }

    auto num_devices = physical_coordinate_to_device_id_.size();
    logical_mesh_shape_ = get_system_mesh_shape(num_devices);
    logical_to_physical_coordinates_ = get_system_mesh_translation_map(num_devices);
    for (const auto& [logical_coordinate, physical_coordinate] : logical_to_physical_coordinates_) {
        logical_to_device_id_.emplace(logical_coordinate, physical_coordinate_to_device_id_.at(physical_coordinate));
    }
}

const MeshShape& SystemMesh::Impl::get_shape() const { return logical_mesh_shape_; }
size_t SystemMesh::Impl::get_num_devices() const {
    auto [num_rows, num_cols] = get_shape();
    return num_rows * num_cols;
}

chip_id_t SystemMesh::Impl::get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const {
    TT_FATAL(
        logical_row_idx < logical_mesh_shape_.num_rows,
        "Row index out of bounds: {} >= {}",
        logical_row_idx,
        logical_mesh_shape_.num_rows);
    TT_FATAL(
        logical_col_idx < logical_mesh_shape_.num_cols,
        "Column index out of bounds: {} >= {}",
        logical_col_idx,
        logical_mesh_shape_.num_cols);
    auto logical_coordinate = Coordinate{logical_row_idx, logical_col_idx};
    return logical_to_device_id_.at(logical_coordinate);
}

std::vector<chip_id_t> SystemMesh::Impl::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    std::vector<chip_id_t> physical_device_ids;
    auto [system_mesh_rows, system_mesh_cols] = get_shape();
    auto [requested_num_rows, requested_num_cols] = config.mesh_shape;
    auto [row_offset, col_offset] = config.offset;

    // First check if total size fits
    TT_FATAL(
        requested_num_rows * requested_num_cols <= system_mesh_rows * system_mesh_cols,
        "Requested submesh is too big: {}x{}",
        requested_num_rows,
        requested_num_cols);

    bool is_single_row_or_column = requested_num_rows == 1 or requested_num_cols == 1;
    if (is_single_row_or_column) {
        TT_FATAL(row_offset == 0 and col_offset == 0, "Row and column offsets unsupported for single row mesh");
        auto line_length = requested_num_rows * requested_num_cols;
        auto line_coords = MeshDeviceView::get_line_coordinates(
            line_length, Coordinate{row_offset, col_offset}, system_mesh_rows, system_mesh_cols);
        for (const auto& logical_coordinate : line_coords) {
            auto physical_device_id = logical_to_device_id_.at(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
        return physical_device_ids;
    }
    bool requires_rotation = requested_num_rows > system_mesh_rows || requested_num_cols > system_mesh_cols;

    if (requires_rotation) {
        bool can_rotate = requested_num_rows <= system_mesh_cols && requested_num_cols <= system_mesh_rows;
        if (can_rotate) {
            // Rotate requested shape; row_offset and col_offset refer to original orientation
            std::swap(requested_num_rows, requested_num_cols);
        } else {
            TT_THROW(
                "User has requested a submesh that is too big and is not rotatable: {}x{} and SystemMesh is {}x{}.",
                requested_num_rows,
                requested_num_cols,
                system_mesh_rows,
                system_mesh_cols);
        }
    } else {
        // If no rotation, check dimensions directly
        TT_FATAL(
            requested_num_rows <= system_mesh_rows && requested_num_cols <= system_mesh_cols,
            "Requested submesh is too big: {}x{} and SystemMesh is {}x{}",
            requested_num_rows,
            requested_num_cols,
            system_mesh_rows,
            system_mesh_cols);
    }

    size_t original_rows = system_mesh_rows;
    size_t original_cols = system_mesh_cols;

    // Check that offsets fit in the original mesh
    TT_FATAL(
        row_offset + requested_num_rows <= original_rows,
        "Row offset + requested rows exceeds mesh size: {} + {} > {}",
        row_offset,
        requested_num_rows,
        original_rows);
    TT_FATAL(
        col_offset + requested_num_cols <= original_cols,
        "Column offset + requested columns exceeds mesh size: {} + {} > {}",
        col_offset,
        requested_num_cols,
        original_cols);

    // Map each submesh coordinate to the original logical coordinates
    for (size_t row = 0; row < requested_num_rows; row++) {
        for (size_t col = 0; col < requested_num_cols; col++) {
            Coordinate logical_coordinate;
            if (requires_rotation) {
                // After swapping requested_num_rows and requested_num_cols,
                // (row, col) now iterate over the rotated shape.
                size_t old_row = row_offset + row;  // top row
                size_t old_col = col_offset + col;  // increasing columns horizontally
                logical_coordinate = Coordinate{old_row, old_col};
            } else {
                logical_coordinate = Coordinate{row + row_offset, col + col_offset};
            }

            TT_FATAL(
                logical_coordinate.row < logical_mesh_shape_.num_rows,
                "Row coordinate out of bounds: {} >= {}",
                logical_coordinate.row,
                logical_mesh_shape_.num_rows);
            TT_FATAL(
                logical_coordinate.col < logical_mesh_shape_.num_cols,
                "Column coordinate out of bounds: {} >= {}",
                logical_coordinate.col,
                logical_mesh_shape_.num_cols);

            auto physical_device_id = logical_to_device_id_.at(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
    }
    return physical_device_ids;
}

void SystemMesh::Impl::register_mesh_device(
    const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<Device*>& devices) {
    std::vector<chip_id_t> physical_device_ids;
    for (auto* device : devices) {
        physical_device_ids.push_back(device->id());
    }
    assigned_mesh_device_devices_.insert({mesh_device->get_mesh_id(), mesh_device});
    assigned_devices_.insert({mesh_device->get_mesh_id(), physical_device_ids});
}

std::vector<chip_id_t> SystemMesh::Impl::request_available_devices(const MeshDeviceConfig& config) const {
    auto [requested_num_rows, requested_num_cols] = config.mesh_shape;
    auto [max_num_rows, max_num_cols] = logical_mesh_shape_;
    auto [row_offset, col_offset] = config.offset;

    log_debug(
        LogMetal,
        "Mapping MeshDevice ({}x{}) with offset: {}, {}",
        requested_num_rows,
        requested_num_cols,
        row_offset,
        col_offset);

    return config.physical_device_ids.empty() ? get_mapped_physical_device_ids(config)
                                              : config.physical_device_ids;
}

SystemMesh::SystemMesh() : pimpl_(std::make_unique<Impl>()) {}
SystemMesh::~SystemMesh() = default;

SystemMesh& SystemMesh::instance() {
    static SystemMesh instance;
    if (!instance.pimpl_->is_system_mesh_initialized()) {
        instance.pimpl_->initialize();
    }
    return instance;
}

chip_id_t SystemMesh::get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const {
    return pimpl_->get_physical_device_id(logical_row_idx, logical_col_idx);
}

const MeshShape& SystemMesh::get_shape() const { return pimpl_->get_shape(); }

size_t SystemMesh::get_num_devices() const { return pimpl_->get_num_devices(); }

void SystemMesh::register_mesh_device(
    const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<Device*>& devices) {
    pimpl_->register_mesh_device(mesh_device, devices);
}

std::vector<chip_id_t> SystemMesh::request_available_devices(const MeshDeviceConfig& config) const {
    return pimpl_->request_available_devices(config);
}

std::vector<chip_id_t> SystemMesh::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    return pimpl_->get_mapped_physical_device_ids(config);
}

Device* MeshDevice::reference_device() const { return devices_.at(0); }

MeshDevice::MeshDevice(const MeshShape& mesh_device_shape, MeshType type, std::weak_ptr<MeshDevice> parent_mesh) :
    mesh_device_shape_(mesh_device_shape),
    type_(type),
    mesh_id_(generate_unique_mesh_id()),
    parent_mesh_(std::move(parent_mesh)) {}

std::shared_ptr<MeshDevice> MeshDevice::create(
    const MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config) {
    auto mesh_device = std::make_shared<MeshDevice>(config.mesh_shape, config.mesh_type);
    mesh_device->initialize(l1_small_size, trace_region_size, num_command_queues, dispatch_core_config, config);

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

    if (offset.row + submesh_shape.num_rows > mesh_device_shape_.num_rows ||
        offset.col + submesh_shape.num_cols > mesh_device_shape_.num_cols) {
        TT_THROW(
            "Submesh ({}x{}) with offset ({}, {}) does not fit within parent mesh ({}x{}).",
            submesh_shape.num_rows,
            submesh_shape.num_cols,
            offset.row,
            offset.col,
            mesh_device_shape_.num_rows,
            mesh_device_shape_.num_cols);
    }

    auto submesh = std::make_shared<MeshDevice>(submesh_shape, type, shared_from_this());
    auto start_coordinate = Coordinate{offset.row, offset.col};
    auto end_coordinate = Coordinate{offset.row + submesh_shape.num_rows - 1, offset.col + submesh_shape.num_cols - 1};
    submesh->view_ = std::make_unique<MeshDeviceView>(*this, start_coordinate, end_coordinate);
    submesh->devices_ = submesh->view_->get_devices();
    SystemMesh::instance().register_mesh_device(submesh, submesh->devices_);
    submeshes_.push_back(submesh);
    log_trace(
        LogMetal,
        "Instantiating submesh {}: {}x{} with offset: {} {}",
        submesh->get_mesh_id(),
        submesh_shape.num_rows,
        submesh_shape.num_cols,
        offset.row,
        offset.col);
    log_trace(LogMetal, "Submesh {} instantiated with {} devices", submesh->get_mesh_id(), submesh->devices);

    return submesh;
}

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::create_submeshes(const MeshShape& submesh_shape, MeshType type) {
    std::vector<std::shared_ptr<MeshDevice>> submeshes;
    for (int row = 0; row < num_rows(); row += submesh_shape.num_rows) {
        for (int col = 0; col < num_cols(); col += submesh_shape.num_cols) {
            auto submesh = create_submesh(submesh_shape, MeshOffset{row, col}, type);
            submeshes.push_back(submesh);
        }
    }
    return submeshes;
}

void MeshDevice::initialize(
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
        devices_.push_back(this->opened_devices_.at(physical_device_id));
    }
    view_ = std::make_unique<MeshDeviceView>(*this);
    system_mesh.register_mesh_device(shared_from_this(), devices_);
}

MeshDevice::~MeshDevice() { close_devices(); }

Device* MeshDevice::get_device_index(size_t logical_device_id) const {
    TT_FATAL(logical_device_id >= 0 and logical_device_id < num_devices(), "Invalid device index");
    return devices_.at(logical_device_id);
}

Device* MeshDevice::get_device(chip_id_t physical_device_id) const {
    for (auto device : devices_) {
        if (device->id() == physical_device_id) {
            return device;
        }
    }
    TT_THROW("Physical Device ID: {} not found in assigned devices", physical_device_id);
}

std::vector<Device*> MeshDevice::get_devices(const std::optional<MeshType>& requested_type) const {
    return view_->get_devices(requested_type.value_or(this->type_));
}

// TODO: Remove this function once we have a proper view interface
Device* MeshDevice::get_device(size_t row_idx, size_t col_idx) const {
    return get_device_index(row_idx * num_cols() + col_idx);
}

const DeviceIds MeshDevice::get_device_ids() const {
    DeviceIds device_ids;
    for (auto device : get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

size_t MeshDevice::num_devices() const { return devices_.size(); }

CoreCoord MeshDevice::compute_with_storage_grid_size() const {
    return reference_device()->compute_with_storage_grid_size();
}

CoreCoord MeshDevice::dram_grid_size() const { return reference_device()->dram_grid_size(); }

tt::ARCH MeshDevice::arch() const { return reference_device()->arch(); }

size_t MeshDevice::num_rows() const { return mesh_device_shape_.num_rows; }

size_t MeshDevice::num_cols() const { return mesh_device_shape_.num_cols; }

MeshShape MeshDevice::shape() const { return mesh_device_shape_; }

void MeshDevice::reshape(const MeshShape& new_shape) {
    TT_FATAL(
        new_shape.num_rows * new_shape.num_cols == num_devices(),
        "New shape must have the same number of devices as current shape");

    std::unordered_map<chip_id_t, size_t> physical_device_id_to_linearized_index;
    for (size_t i = 0; i < num_devices(); i++) {
        physical_device_id_to_linearized_index[devices_[i]->id()] = i;
    }

    // From an MxN mesh, we can always reduce rank to a 1xM*N Line mesh.
    // However, going from a Line mesh to an MxN mesh is not always possible.
    if (new_shape.num_rows != 1 and new_shape.num_cols != 1) {
        auto new_physical_device_ids = SystemMesh::instance().request_available_devices(MeshDeviceConfig{new_shape});

        for (size_t i = 0; i < new_physical_device_ids.size(); i++) {
            if (physical_device_id_to_linearized_index.find(new_physical_device_ids[i]) ==
                physical_device_id_to_linearized_index.end()) {
                TT_THROW(
                    "User has requested a reshape of the MeshDevice to shape: {}x{}, but it is not possible to form a "
                    "physically connected mesh of {}x{} grid with the opened devices from the original shape: {}x{}.",
                    new_shape.num_rows,
                    new_shape.num_cols,
                    new_shape.num_rows,
                    new_shape.num_cols,
                    num_rows(),
                    num_cols());
            }
        }
    }

    mesh_device_shape_ = new_shape;
    view_ = std::make_unique<MeshDeviceView>(*this);
}

void MeshDevice::close_devices() {
    for (const auto& submesh : submeshes_) {
        submesh->close_devices();
    }
    if (!opened_devices_.empty()) {
        tt::tt_metal::detail::CloseDevices(opened_devices_);
        opened_devices_.clear();
    }
    submeshes_.clear();
    parent_mesh_.reset();
    devices_.clear();
    view_.reset();
}

std::string MeshDevice::to_string() const {
    return fmt::format("MeshDevice({}x{} grid, {} devices)", num_rows(), this->num_cols(), this->num_devices());
}

const MeshDeviceView& MeshDevice::get_view() const {
    TT_FATAL(view_, "MeshDeviceView is not initialized");
    return *view_;
}

MeshDeviceID MeshDevice::get_mesh_id() const { return mesh_id_; }

bool MeshDevice::is_parent_mesh() const { return parent_mesh_.expired(); }

std::vector<std::shared_ptr<MeshDevice>> MeshDevice::get_submeshes() const { return submeshes_; }

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device) { return os << mesh_device.to_string(); }

void MeshDevice::enable_async(bool enable) {
    for (auto device : devices_) {
        device->enable_async(enable);
    }
}

void MeshDevice::enable_program_cache() {
    for (auto device : devices_) {
        device->enable_program_cache();
    }
}

void MeshDevice::disable_and_clear_program_cache() {
    for (auto device : devices_) {
        device->disable_and_clear_program_cache();
    }
}

size_t MeshDevice::num_program_cache_entries() const {
    size_t total_entries = 0;
    for (auto device : devices_) {
        total_entries += device->num_program_cache_entries();
    }
    return total_entries;
}

MeshSubDeviceManagerId MeshDevice::create_sub_device_manager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    MeshSubDeviceManagerId mesh_sub_device_manager_id(*this);
    for (uint32_t i = 0; i < num_devices(); i++) {
        auto* device = devices_[i];
        auto& sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work([device, sub_devices, local_l1_size, &sub_device_manager_id]() {
            sub_device_manager_id = device->create_sub_device_manager(sub_devices, local_l1_size);
        });
    }
    for (auto* device : devices_) {
        device->synchronize();
    }
    return mesh_sub_device_manager_id;
}

std::tuple<MeshSubDeviceManagerId, SubDeviceId> MeshDevice::create_sub_device_manager_with_fabric(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    MeshSubDeviceManagerId mesh_sub_device_manager_id(*this);
    SubDeviceId fabric_sub_device_id;
    for (uint32_t i = 0; i < num_devices(); i++) {
        auto* device = devices_[i];
        auto& sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        // All fabric sub-device ids will be the same, since all managers are created with the same sub_devices input
        device->push_work([device, sub_devices, local_l1_size, &sub_device_manager_id, &fabric_sub_device_id]() {
            std::tie(sub_device_manager_id, fabric_sub_device_id) =
                device->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
        });
    }
    for (auto* device : devices_) {
        device->synchronize();
    }
    return {mesh_sub_device_manager_id, fabric_sub_device_id};
}

void MeshDevice::load_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id) {
    for (uint32_t i = 0; i < num_devices(); i++) {
        auto* device = devices_[i];
        auto sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work(
            [device, sub_device_manager_id]() { device->load_sub_device_manager(sub_device_manager_id); });
    }
}
void MeshDevice::clear_loaded_sub_device_manager() {
    for (auto* device : devices_) {
        device->push_work([device]() { device->clear_loaded_sub_device_manager(); });
    }
}
void MeshDevice::remove_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id) {
    for (uint32_t i = 0; i < num_devices(); i++) {
        auto* device = devices_[i];
        auto sub_device_manager_id = mesh_sub_device_manager_id.sub_device_manager_ids[i];
        device->push_work(
            [device, sub_device_manager_id]() { device->remove_sub_device_manager(sub_device_manager_id); });
    }
}

MeshSubDeviceManagerId::MeshSubDeviceManagerId(const MeshDevice& mesh_device) {
    sub_device_manager_ids.resize(mesh_device.num_devices());
}

int MeshDevice::num_dram_channels() const {
    return reference_device()->num_dram_channels() * this->num_devices();
}

allocator::Statistics MeshDevice::get_memory_allocation_statistics(
    const BufferType& buffer_type, SubDeviceId sub_device_id) const {
    // With current implementation, we assume that all devices have the same memory allocation statistics.
    // This will be made more explicit in the future to have lock-step allocation across devices.
    // Right now, we just return the statistics of the first device.
    return reference_device()->get_memory_allocation_statistics(buffer_type, sub_device_id);
}

}  // namespace tt::tt_metal::distributed
