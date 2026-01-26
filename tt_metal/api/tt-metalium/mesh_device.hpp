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
#include <umd/device/types/arch.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {
class Allocator;
class CommandQueue;
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

private:
    MeshDevice() = default;

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
    ChipId build_id() const override;
    uint8_t num_hw_cqs() const override;
    bool is_initialized() const override;

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;

    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const override;
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const override;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const override;
    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const override;
    std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const override;
    std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const override;
    bool is_inactive_ethernet_core(CoreCoord logical_core) const override;
    uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) override;
    CoreCoord compute_with_storage_grid_size() const override;
    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl() const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl(SubDeviceId sub_device_id) const override;
    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const override;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const override;
    uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const override;
    const std::set<CoreCoord>& ethernet_cores() const override;
    const std::set<CoreCoord>& storage_only_cores() const override;
    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;
    SystemMemoryManager& sysmem_manager() override;
    CommandQueue& command_queue(std::optional<uint8_t> cq_id = std::nullopt) override;

    // MeshTrace Internal APIs - these should be used to deprecate the single device backed trace APIs
    // If cq_id is not provided, the current command queue is returned from the current thread
    MeshTraceId begin_mesh_trace(uint8_t cq_id);
    void begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking);
    void release_mesh_trace(const MeshTraceId& trace_id);
    std::shared_ptr<MeshTraceBuffer> get_mesh_trace(const MeshTraceId& trace_id);
    uint32_t get_trace_buffers_size() const override;
    void set_trace_buffers_size(uint32_t size) override;

    // Initialization APIs
    bool initialize(
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;
    bool compile_fabric() override;
    void configure_fabric() override;
    void init_fabric() override;
    bool close() override;
    void enable_program_cache() override;
    void clear_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override;
    std::size_t num_program_cache_entries() override;
    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;
    HalMemType get_mem_type_of_core(CoreCoord virtual_core) const override;
    bool has_noc_mcast_txns(SubDeviceId sub_device_id) const override;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const override;
    uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const override;
    SubDeviceManagerId get_active_sub_device_manager_id() const override;
    SubDeviceManagerId get_default_sub_device_manager_id() const override;
    SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void clear_loaded_sub_device_manager() override;
    CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const override;
    const std::vector<SubDeviceId>& get_sub_device_ids() const override;
    const std::vector<SubDeviceId>& get_sub_device_stall_group() const override;
    void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) override;
    void reset_sub_device_stall_group() override;
    uint32_t num_sub_devices() const override;
    bool is_mmio_capable() const override;
    std::shared_ptr<distributed::MeshDevice> get_mesh_device() override;

    // A MeshDevice is a collection of devices arranged in a 2D grid.
    // The type parameter allows the caller to specify how to linearize the devices in the mesh.

    // Returns the devices in the mesh in row-major order.
    std::vector<IDevice*> get_devices() const;
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

    std::string to_string() const;
    bool is_parent_mesh() const;

    const std::shared_ptr<MeshDevice>& get_parent_mesh() const;
    std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;

    /**
     * @brief Synchronize with all devices derived from this mesh (including submeshes).
     *
     * Blocks until all in-flight work enqueued on every submesh derived from this mesh has completed. Use this to
     * insert a barrier between phases that use overlapping submeshes on the same physical devices. After this call
     * returns, it is safe to enqueue new work on this mesh or any submesh derived from this mesh that may overlap with
     * submeshes that were previously active. All submeshes must be using the default subdevice manager when this is
     * called.
     */
    void quiesce_devices();

    std::shared_ptr<MeshDevice> create_submesh(
        const MeshShape& submesh_shape, const std::optional<MeshCoordinate>& offset = std::nullopt);

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(const MeshShape& submesh_shape);

    // This method will get removed once in favour of the ones in IDevice* and TT-Mesh bringup
    // These are prefixed with "mesh_" to avoid conflicts with the IDevice* methods
    // If cq_id is not provided, the current command queue is returned from the current thread
    MeshCommandQueue& mesh_command_queue(std::optional<uint8_t> cq_id = std::nullopt) const;

    // Currently expose users to the dispatch thread pool through the MeshDevice
    void enqueue_to_thread_pool(std::function<void()>&& f);
    void wait_for_thread_pool();
    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
    static std::shared_ptr<MeshDevice> create_unit_mesh(
        int device_id,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
    static std::map<int, std::shared_ptr<MeshDevice>> create_unit_meshes(
        const std::vector<int>& device_ids,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    // Only for internal and testing purposes
    const MeshDeviceImpl& impl() const { return *pimpl_; }
    MeshDeviceImpl& impl() { return *pimpl_; }
};

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device);

}  // namespace distributed

}  // namespace tt::tt_metal
