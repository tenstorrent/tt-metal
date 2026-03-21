// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/sub_device_types.hpp>
// UMD: re-exports tt::ARCH (used in MeshDevice::arch return type).
#include <umd/device/types/arch.hpp>
// UMD: re-exports CoreType (used in MeshDevice::virtual_core_from_logical_core parameter).
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {
class Allocator;
class MetalEnv;
class SubDevice;
class SystemMemoryManager;

namespace program_cache::detail {
struct ProgramCache;
}  // namespace program_cache::detail

}  // namespace tt::tt_metal

namespace tt::tt_fabric {
class FabricNodeId;
}
namespace tt::tt_metal {

class SubDeviceManagerTracker;
class ThreadPool;
struct TraceDescriptor;

namespace distributed {

class MeshCommandQueue;
class MeshDeviceView;
struct MeshTraceBuffer;
class MeshCommandQueueBase;
class MeshDeviceImpl;

namespace multihost {
class DistributedContext;
}

using DeviceIds = std::vector<int>;

class MeshDevice : public IDevice, public std::enable_shared_from_this<MeshDevice> {
    friend class MeshDeviceImpl;
    friend class tt::tt_metal::MetalEnv;

private:
    MeshDevice() = default;
    // [[Experimental]] Creates a MeshDevice that uses the given MetalEnv instance.
    // This is used by MetalEnv::create_mesh_device and MetalEnv::create_unit_mesh_device.
    explicit MeshDevice(MetalEnv& metal_env);

    std::unique_ptr<MeshDeviceImpl> pimpl_;

public:
    ~MeshDevice() override;

    MeshDevice(const MeshDevice&) = delete;
    MeshDevice& operator=(const MeshDevice&) = delete;

    MeshDevice(MeshDevice&&) = delete;
    MeshDevice& operator=(MeshDevice&&) = delete;

    // IDevice interface implementation
    tt::ARCH arch() const override;
    int id() const override;

    // Access to internal methods (delegates to pimpl_)
    IDeviceInternal& device_internal() override;
    const IDeviceInternal& device_internal() const override;

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;
    // Returns the AI clock frequency in MHz for this device.
    // This value is queried from the actual hardware via the cluster API
    // and reflects the device's current operating frequency.
    int get_clock_rate_mhz() const override;

    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;

    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;

    CoreCoord compute_with_storage_grid_size() const override;
    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        ttsl::Span<const SubDeviceId> sub_device_ids) const override;
    const std::set<CoreCoord>& storage_only_cores() const override;

    // MeshTrace Internal APIs - these should be used to deprecate the single device backed trace APIs
    // If cq_id is not provided, the current command queue is returned from the current thread
    MeshTraceId begin_mesh_trace(uint8_t cq_id);
    void begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking);
    void release_mesh_trace(const MeshTraceId& trace_id);

    bool close() override;
    void enable_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override;

    SubDeviceManagerId get_active_sub_device_manager_id() const override;
    SubDeviceManagerId get_default_sub_device_manager_id() const override;
    const std::vector<SubDeviceId>& get_sub_device_ids() const override;

    bool is_mmio_capable() const override;

    // Program cache management (public API)
    void clear_program_cache() override;
    void disable_and_clear_program_cache() override;
    std::size_t num_program_cache_entries() override;

    // Sub-device management (public API)
    SubDeviceManagerId create_sub_device_manager(
        ttsl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void clear_loaded_sub_device_manager() override;
    void set_sub_device_stall_group(ttsl::Span<const SubDeviceId> sub_device_ids) override;
    void reset_sub_device_stall_group() override;

    // Ethernet connectivity (public API)
    std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const override;

    // A MeshDevice is a collection of devices arranged in a 2D grid.
    // The type parameter allows the caller to specify how to linearize the devices in the mesh.

    [[deprecated(
        "Deprecated, retrieving physical devices can fail in distributed contexts. This will be removed after "
        "28-02-2026.")]]
    IDevice* get_device(ChipId physical_device_id) const;
    [[deprecated(
        "Deprecated, retrieving physical devices can fail in distributed contexts. This will be removed after "
        "28-02-2026.")]]
    IDevice* get_device(const MeshCoordinate& coord) const;
    tt_fabric::FabricNodeId get_fabric_node_id(const MeshCoordinate& coord) const;

    DeviceIds get_device_ids() const;

    size_t num_devices() const;

    // The following methods assume 2D mesh, and throw if the mesh is not 2D.
    // TODO: #17477 - Remove the methods that assume 2D mesh.
    size_t num_rows() const;
    size_t num_cols() const;
    IDevice* get_device(size_t row_idx, size_t col_idx) const;

    // Returns true if the coordinate is local to this mesh device.
    // Throws if the coordinate is out of bounds of this mesh device.
    [[deprecated(
        "Deprecated, is_local should be avoided as it is likely to cause issues in distributed contexts. This will be "
        "removed after 28-02-2026.")]]
    bool is_local(const MeshCoordinate& coord) const;

    const MeshShape& shape() const;

    // Reshapes the logical mesh and re-maps the physical devices to the new logical coordinates.
    // Reshaping Rules:
    // 1. The old_shape volume must equal the new_shape volume (i.e. number of devices must remain constant)
    // 2. Line-to-Line Reshaping (when either dimension is 1):
    //    - Always possible between 1xN and Nx1 shapes (e.g.: 1x8 <-> 8x1)
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
    const MeshDeviceView& get_view() const;

    // Returns the system mesh ID from the underlying view.
    // This ID is programmed in the Mesh Graph Descriptor that a user provides for a Multi-Mesh Topology.
    // This value defaults to zero, for all workloads running within a single process.
    // For distributed Multi-Mesh workloads, this value represents which Fabric Mesh the MeshDevice belongs to,
    // in the Logical Graph.
    // TODO: https://github.com/tenstorrent/tt-metal/issues/38385
    uint32_t get_system_mesh_id() const;

    std::string to_string() const;

    const std::shared_ptr<MeshDevice>& get_parent_mesh() const;
    std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;

    std::shared_ptr<MeshDevice> create_submesh(
        const MeshShape& submesh_shape, const std::optional<MeshCoordinate>& offset = std::nullopt);

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(const MeshShape& submesh_shape);

    // This method will get removed once in favour of the ones in IDevice* and TT-Mesh bringup
    // These are prefixed with "mesh_" to avoid conflicts with the IDevice* methods
    // If cq_id is not provided, the current command queue is returned from the current thread
    MeshCommandQueue& mesh_command_queue(std::optional<uint8_t> cq_id = std::nullopt) const;

    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
    static std::shared_ptr<MeshDevice> create_unit_mesh(
        int device_id,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
    static std::map<int, std::shared_ptr<MeshDevice>> create_unit_meshes(
        const std::vector<int>& device_ids,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    // Only for internal and testing purposes
    const MeshDeviceImpl& impl() const { return *pimpl_; }
    MeshDeviceImpl& impl() { return *pimpl_; }
};

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device);

}  // namespace distributed

}  // namespace tt::tt_metal
