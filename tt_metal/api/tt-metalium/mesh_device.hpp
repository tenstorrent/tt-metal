// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
#include <umd/device/types/arch.h>

enum class CoreType;
namespace tt {
namespace tt_metal {
class Allocator;
class CommandQueue;
class SubDevice;
class SystemMemoryManager;
namespace program_cache {
namespace detail {
struct ProgramCache;
}  // namespace detail
}  // namespace program_cache
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_fabric {
class FabricNodeId;
}
namespace tt::tt_metal {

class SubDeviceManagerTracker;
class ThreadPool;
class TraceDescriptor;

namespace distributed {

class MeshCommandQueue;
class MeshDeviceView;
class MeshTraceBuffer;

using DeviceIds = std::vector<int>;

class MeshDevice : public IDevice, public std::enable_shared_from_this<MeshDevice> {
private:
    // Resource management class / RAII wrapper for *physical devices* of the mesh
    class ScopedDevices {
    private:
        std::map<chip_id_t, IDevice*> opened_devices_;
        std::vector<IDevice*> devices_;

    public:
        // Constructor acquires physical resources
        ScopedDevices(
            size_t l1_small_size,
            size_t trace_region_size,
            size_t num_command_queues,
            size_t worker_l1_size,
            const DispatchCoreConfig& dispatch_core_config,
            const MeshDeviceConfig& config);
        ScopedDevices(
            const std::vector<int>& device_ids,
            size_t l1_small_size,
            size_t trace_region_size,
            size_t num_command_queues,
            size_t worker_l1_size,
            const DispatchCoreConfig& dispatch_core_config);

        // Destructor releases physical resources
        ~ScopedDevices();
        ScopedDevices(const ScopedDevices&) = delete;
        ScopedDevices& operator=(const ScopedDevices&) = delete;

        // Returns the list of devices opened by the root mesh device (i.e. not submeshes).
        const std::vector<IDevice*>& root_devices() const;
    };
    std::shared_ptr<ScopedDevices> scoped_devices_;
    int mesh_id_;
    std::unique_ptr<MeshDeviceView> view_;
    // Submesh keeps the parent mesh alive. Parent_mesh_ is null if the current mesh is the parent mesh.
    std::shared_ptr<MeshDevice> parent_mesh_;
    std::vector<std::weak_ptr<MeshDevice>> submeshes_;

    tt::stl::SmallVector<std::unique_ptr<MeshCommandQueue>> mesh_command_queues_;

    std::unique_ptr<SubDeviceManagerTracker> sub_device_manager_tracker_;
    std::unordered_map<MeshTraceId, std::shared_ptr<MeshTraceBuffer>> trace_buffer_pool_;
    uint32_t trace_buffers_size_ = 0;
    uint32_t max_num_eth_cores_ = 0;
    std::shared_ptr<ThreadPool> dispatch_thread_pool_;
    std::shared_ptr<ThreadPool> reader_thread_pool_;
    // Num Virtual Eth Cores == Max Number of Eth Cores across all opened devices (Issue #19729)
    std::size_t num_virtual_eth_cores_ = 0;
    std::unique_ptr<program_cache::detail::ProgramCache> program_cache_;
    // This is a reference device used to query properties that are the same for all devices in the mesh.
    IDevice* reference_device() const;

    // Returns the devices in row-major order for the new mesh shape
    std::vector<IDevice*> get_row_major_devices(const MeshShape& new_shape) const;

    std::shared_ptr<MeshTraceBuffer>& create_mesh_trace(const MeshTraceId& trace_id);

public:
    MeshDevice(
        std::shared_ptr<ScopedDevices> scoped_devices,
        std::unique_ptr<MeshDeviceView> mesh_device_view,
        std::shared_ptr<MeshDevice> parent_mesh = {});
    ~MeshDevice() override;

    MeshDevice(const MeshDevice&) = delete;
    MeshDevice& operator=(const MeshDevice&) = delete;

    MeshDevice(MeshDevice&&) = delete;
    MeshDevice& operator=(MeshDevice&&) = delete;

    // IDevice interface implementation
    tt::ARCH arch() const override;
    int id() const override;
    chip_id_t build_id() const override;
    uint8_t num_hw_cqs() const override;
    bool is_initialized() const override;

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;

    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>&logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const override;
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const override;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const override;
    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const override;
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const override;
    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const override;
    bool is_inactive_ethernet_core(CoreCoord logical_core) const override;
    uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) override;
    CoreCoord compute_with_storage_grid_size() const override;
    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;
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
    CommandQueue& command_queue(size_t cq_id = 0) override;

    // Trace APIs
    void begin_trace(uint8_t cq_id, uint32_t tid) override;
    void end_trace(uint8_t cq_id, uint32_t tid) override;

    // TODO: `block_on_worker_thread` can be removed once we remove multi-threaded async dispatch
    void replay_trace(uint8_t cq_id, uint32_t tid, bool block_on_device, bool block_on_worker_thread) override;
    void release_trace(uint32_t tid) override;
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) override;

    // MeshTrace Internal APIs - these should be used to deprecate the single device backed trace APIs
    void begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id);
    void replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking);
    void release_mesh_trace(const MeshTraceId& trace_id);
    std::shared_ptr<MeshTraceBuffer> get_mesh_trace(const MeshTraceId& trace_id);
    uint32_t get_trace_buffers_size() const override;
    void set_trace_buffers_size(uint32_t size) override;

    // Light Metal
    void load_trace(uint8_t cq_id, uint32_t trace_id, const TraceDescriptor& trace_desc) override;

    bool using_slow_dispatch() const override;
    bool using_fast_dispatch() const override;

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
    SubDeviceManagerId create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
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
    IDevice* get_device(chip_id_t physical_device_id) const;
    IDevice* get_device(const MeshCoordinate& coord) const;
    tt_fabric::FabricNodeId get_device_fabric_node_id(const MeshCoordinate& coord) const;

    DeviceIds get_device_ids() const;

    size_t num_devices() const;

    // The following methods assume 2D mesh, and throw if the mesh is not 2D.
    // TODO: #17477 - Remove the methods that assume 2D mesh.
    size_t num_rows() const;
    size_t num_cols() const;
    IDevice* get_device(size_t row_idx, size_t col_idx) const;

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

    std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;

    std::shared_ptr<MeshDevice> create_submesh(
        const MeshShape& submesh_shape, const std::optional<MeshCoordinate>& offset = std::nullopt);

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(const MeshShape& submesh_shape);

    // This method will get removed once in favour of the ones in IDevice* and TT-Mesh bringup
    // These are prefixed with "mesh_" to avoid conflicts with the IDevice* methods
    MeshCommandQueue& mesh_command_queue(std::size_t cq_id = 0) const;

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

    // Returns the offset of this host's portion of the mesh within the global distributed mesh.
    // For single-host meshes, this returns (0, 0).
    MeshCoordinate local_offset() const;

    // Returns the shape of the mesh portion managed by this host.
    // For single-host meshes, this equals the global mesh shape.
    MeshShape local_shape() const;

    // Checks if a global coordinate is managed by this host.
    // Returns true if the coordinate falls within this host's local mesh bounds.
    bool is_local_coordinate(const MeshCoordinate& coord) const;
};

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device);

}  // namespace distributed

}  // namespace tt::tt_metal
