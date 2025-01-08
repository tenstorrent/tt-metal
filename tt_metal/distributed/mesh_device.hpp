// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/device.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/tt_stl/span.hpp"

namespace tt::tt_metal::distributed {

using DeviceIds = std::vector<int>;
using MeshDeviceID = size_t;
struct MeshOffset {
    size_t row = 0;
    size_t col = 0;
};

class MeshCommandQueue;

class MeshDeviceView;

struct MeshSubDeviceManagerId;

struct MeshDeviceConfig {
    MeshShape mesh_shape;
    MeshOffset offset;
    std::vector<chip_id_t> physical_device_ids;
    MeshType mesh_type;

    MeshDeviceConfig(const MeshShape& mesh_shape, MeshType mesh_type) :
        mesh_shape(mesh_shape),
        offset(MeshOffset{0, 0}),
        physical_device_ids(std::vector<chip_id_t>()),
        mesh_type(mesh_type) {}

    MeshDeviceConfig(
        const MeshShape& mesh_shape,
        const MeshOffset& offset = MeshOffset{0, 0},
        const std::vector<chip_id_t>& physical_device_ids = {},
        MeshType mesh_type = MeshType::RowMajor) :
        mesh_shape(mesh_shape), offset(offset), physical_device_ids(physical_device_ids), mesh_type(mesh_type) {}
};

// SystemMesh creates a virtualization over the physical devices in the system.
// It creates a logical 2D-mesh of devices and manages the mapping between logical and physical device coordinates.
// It serves as a query interface between the logical 2D coordinates to physical device IDs.
class SystemMesh {
   private:
    friend class MeshDevice;
    class Impl;  // Forward declaration only
    std::unique_ptr<Impl> pimpl_;
    SystemMesh();
    ~SystemMesh();

    std::vector<chip_id_t> request_available_devices(const MeshDeviceConfig& config) const;
    void register_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<IDevice*>& devices);

public:
    static SystemMesh& instance();
    SystemMesh(const SystemMesh &) = delete;
    SystemMesh &operator=(const SystemMesh &) = delete;
    SystemMesh(SystemMesh &&) = delete;
    SystemMesh &operator=(SystemMesh &&) = delete;

    // Get the shape of the logical mesh
    const MeshShape& get_shape() const;
    size_t get_num_devices() const;

    // Gets the physical device ID for a given logical row and column index
    chip_id_t get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const;

    // Get the physical device IDs mapped to a MeshDevice
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig &config) const;
};

class MeshDevice : public std::enable_shared_from_this<MeshDevice> {
private:
    MeshDeviceID mesh_id;
    MeshShape mesh_device_shape;
    MeshType type;
    std::unique_ptr<MeshDeviceView> view;
    std::map<chip_id_t, IDevice*> opened_devices;
    std::vector<IDevice*> devices;
    std::vector<std::shared_ptr<MeshDevice>>
        submeshes;  // Parent owns submeshes and is responsible for their destruction
    std::weak_ptr<MeshDevice> parent_mesh;               // Submesh created with reference to parent mesh
    std::unique_ptr<MeshCommandQueue> mesh_command_queue_;

    void initialize(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t num_command_queues,
        const DispatchCoreConfig& dispatch_core_config,
        const MeshDeviceConfig& config);

    // This is a reference device used to query properties that are the same for all devices in the mesh.
    IDevice* reference_device() const;

public:
    MeshDevice(const MeshShape& mesh_device_shape, MeshType type, std::weak_ptr<MeshDevice> parent_mesh = {});
    ~MeshDevice();

    MeshDevice(const MeshDevice&) = delete;
    MeshDevice& operator=(const MeshDevice&) = delete;

    MeshDevice(MeshDevice&&) = delete;
    MeshDevice& operator=(MeshDevice&&) = delete;

    // A MeshDevice is a collection of devices arranged in a 2D grid.
    // The type parameter allows the caller to specify how to linearize the devices in the mesh.
    // If type is not provided, the default behavior is to return the devices based on the MeshType of the MeshDevice.
    std::vector<IDevice*> get_devices(const std::optional<MeshType>& type = std::nullopt) const;
    IDevice* get_device_index(size_t logical_device_id) const;
    IDevice* get_device(chip_id_t physical_device_id) const;
    IDevice* get_device(size_t row_idx, size_t col_idx) const;

    MeshCommandQueue& command_queue();
    const DeviceIds get_device_ids() const;

    size_t num_devices() const;
    size_t num_rows() const;
    size_t num_cols() const;
    MeshShape shape() const;

    // Reshapes the logical mesh and re-maps the physical devices to the new logical coordinates.
    // Reshaping Rules:
    // 1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
    // 2. Line-to-Line Reshaping (when either dimension is 1):
    //    - Always possible between 1xN and Nx1 shapes (e.g.: 1x8 <-> 8x1
    // 3. Grid-to-Grid Reshaping:
    //    - Only possible if the devices can form a connected physical mesh in the new shape
    //    - Must maintain physical connectivity between adjacent devices
    // 4. Line-to-Grid Reshaping:
    //    - Only possible if the physical devices can form a connected physical mesh in the new shape
    //    - Example: 1x8 -> 2x4 is possible only if physical mesh permits a 2x4 configuration
    //
    // @throws std::runtime_error if any of the following constraints are not met:
    // 1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
    // 2. For Grid-to-Grid or Line-to-Grid reshaping: physical connectivity must be possible with current devices
    void reshape(const MeshShape& new_shape);
    void close();
    const MeshDeviceView& get_view() const;

    std::string to_string() const;
    MeshDeviceID id() const;
    bool is_parent_mesh() const;

    std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;

    std::shared_ptr<MeshDevice> create_submesh(
        const MeshShape& submesh_shape,
        const MeshOffset& offset = MeshOffset{0, 0},
        MeshType type = MeshType::RowMajor);

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(
        const MeshShape& submesh_shape, MeshType type = MeshType::RowMajor);

    MeshSubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    // TODO #16526: Temporary api until migration to actual fabric is complete
    std::tuple<MeshSubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    void load_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id);
    void clear_loaded_sub_device_manager();
    void remove_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id);
    // TODO #16492: Add get_sub_device_stall_group once MeshDevice is no longer just a collection of single Devices
    // and the MeshDevice has a single SubDeviceManager responsible for all Devices.
    void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids);
    void reset_sub_device_stall_group();

    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{});

    // Device API Queries (API contract with Device class to be supported in future)
    CoreCoord compute_with_storage_grid_size() const;
    CoreCoord dram_grid_size() const;

    tt::ARCH arch() const;
    void enable_async(bool enable);
    void enable_program_cache();
    void disable_and_clear_program_cache();

    size_t num_program_cache_entries() const;

    int num_dram_channels() const;
    allocator::Statistics get_memory_allocation_statistics(
        const BufferType& buffer_type, SubDeviceId sub_device_id = SubDeviceId{0}) const;

    bool using_fast_dispatch();
};

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device);

// TODO: This will be removed once we have DistributedDevice
// Currently required since each device manages its own sub-device manager ids
struct MeshSubDeviceManagerId {
    MeshSubDeviceManagerId(const MeshDevice& mesh_device);

    std::vector<SubDeviceManagerId> sub_device_manager_ids;
};

}  // namespace tt::tt_metal::distributed
