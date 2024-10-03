// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/mesh_device_view.hpp"

namespace tt::tt_metal {

using DeviceIds = std::vector<int>;
using MeshDeviceID = size_t;
using MeshOffset = std::pair<size_t, size_t>;
class MeshDeviceView;

struct MeshDeviceConfig {
    MeshShape mesh_shape;
    MeshOffset offset;
    std::vector<chip_id_t> physical_device_ids;
    MeshType mesh_type;

    MeshDeviceConfig(
        const MeshShape &mesh_shape,
        MeshType mesh_type) :
        mesh_shape(mesh_shape),
        offset(MeshOffset{0, 0}),
        physical_device_ids(std::vector<chip_id_t>()),
        mesh_type(mesh_type) {}

    MeshDeviceConfig(
        const MeshShape &mesh_shape,
        const MeshOffset &offset = MeshOffset{0, 0},
        const std::vector<chip_id_t> &physical_device_ids = {},
        MeshType mesh_type = MeshType::RowMajor) :
        mesh_shape(mesh_shape),
        offset(offset),
        physical_device_ids(physical_device_ids),
        mesh_type(mesh_type) {}
};

// SystemMesh creates a virtualization over the physical devices in the system.
// It creates a logical 2D-mesh of devices and manages the mapping between logical and physical device coordinates.
// It is responsible for the assignment of devices in a MeshDevice to physical devices, and the creation and deletion of
// device resources.
class SystemMesh {
   private:
    using LogicalCoordinate = Coordinate;
    using PhysicalCoordinate = eth_coord_t;

    // Keep track of the devices that were opened so we can close them later. We shouldn't
    // to keep track of this but DevicePool seems to open all devices associated with an MMIO device id
    std::unordered_map<MeshDeviceID, std::map<chip_id_t, Device*>> opened_devices;
    std::unordered_map<MeshDeviceID, std::vector<chip_id_t>> assigned_devices;
    std::unordered_map<MeshDeviceID, std::shared_ptr<MeshDevice>> assigned_mesh_device_devices;
    std::unordered_map<chip_id_t, Device *> assigned_physical_id_to_device;

    // Logical mesh shape and coordinates
    MeshShape logical_mesh_shape;
    std::unordered_map<LogicalCoordinate, PhysicalCoordinate> logical_to_physical_coordinates;

    // Handling of physical coordinates
    std::unordered_map<PhysicalCoordinate, chip_id_t> physical_coordinate_to_device_id;
    std::unordered_map<chip_id_t, PhysicalCoordinate> physical_device_id_to_coordinate;

    SystemMesh() = default;
    SystemMesh(const SystemMesh &) = delete;
    SystemMesh &operator=(const SystemMesh &) = delete;
    SystemMesh(SystemMesh &&) = delete;
    SystemMesh &operator=(SystemMesh &&) = delete;

    static MeshShape get_system_mesh_shape(size_t system_num_devices);
    static std::unordered_map<LogicalCoordinate, PhysicalCoordinate> get_system_mesh_translation_map(
        size_t system_num_devices);

    bool is_system_mesh_initialized() const;

   public:
    static SystemMesh &instance();

    void initialize();

    // Return the shape of the logical mesh
    const MeshShape &get_shape() const;
    size_t get_num_devices() const;

    // Get the physical device IDs mapped to a MeshDevice
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig &config) const;
    void register_mesh_device(const std::shared_ptr<MeshDevice> &mesh_device, const std::vector<Device*>& devices);

    // Map MeshDevice to physical devices
    std::vector<Device *> map_mesh_device(
        std::shared_ptr<MeshDevice> mesh_device,
        size_t num_command_queues,
        size_t l1_small_size,
        size_t trace_region_size,
        DispatchCoreType dispatch_core_type,
        const MeshDeviceConfig &config);

    // Unmap MeshDevice, releasing the associated physical devices.
    void unmap_mesh_device(const MeshDevice* mesh_device);
    std::shared_ptr<MeshDevice> get_mesh_device(const std::vector<chip_id_t>& physical_device_ids);
    Device* get_device(const chip_id_t physical_device_id) const;
};

class MeshDevice : public std::enable_shared_from_this<MeshDevice> {
  private:
    MeshDeviceID mesh_id;
    MeshShape mesh_device_shape;
    MeshType type;
    std::shared_ptr<MeshDeviceView> primary_view;
    std::vector<Device *> devices;
    std::vector<std::shared_ptr<MeshDevice>> submeshes; // Parent owns submeshes and responsible fortheir destruction
    std::weak_ptr<MeshDevice> parent_mesh; // Submesh created with reference to parent mesh

    void initialize(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t num_command_queues,
        DispatchCoreType dispatch_core_type,
        const MeshDeviceConfig &config);

   public:
    MeshDevice(const MeshShape &mesh_device_shape, MeshType type, std::weak_ptr<MeshDevice> parent_mesh = {});
    ~MeshDevice();

    MeshDevice(const MeshDevice &) = delete;
    MeshDevice &operator=(const MeshDevice &) = delete;

    MeshDevice(MeshDevice &&) = delete;
    MeshDevice &operator=(MeshDevice &&) = delete;

    std::vector<Device *> get_devices() const;
    Device *get_device_index(size_t logical_device_id) const;
    Device *get_device(chip_id_t physical_device_id) const;
    Device *get_device(size_t row_idx, size_t col_idx) const;

    const DeviceIds get_device_ids() const;

    size_t num_devices() const;
    size_t num_rows() const;
    size_t num_cols() const;
    MeshShape shape() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord dram_grid_size() const;

    tt::ARCH arch() const;

    void close_devices();
    std::shared_ptr<const MeshDeviceView> get_view() const;
    std::shared_ptr<MeshDeviceView> get_view();

    std::string to_string() const;
    MeshDeviceID get_mesh_id() const;
    bool is_parent_mesh() const;

    std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;

    std::shared_ptr<MeshDevice> create_submesh(
        const MeshShape &submesh_shape,
        const MeshOffset &offset = MeshOffset{0, 0},
        MeshType type = MeshType::RowMajor);

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(
        const MeshShape &submesh_shape,
        MeshType type = MeshType::RowMajor);

    static std::shared_ptr<MeshDevice> fetch_mesh_device(const std::vector<Device*>& devices);
    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig &config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER);
};

std::ostream &operator<<(std::ostream &os, const MeshDevice &mesh_device);
bool validate_worker_modes(const std::vector<Device *> &workers);
std::vector<int> get_t3k_physical_device_ids_ring();

}  // namespace tt::tt_metal
