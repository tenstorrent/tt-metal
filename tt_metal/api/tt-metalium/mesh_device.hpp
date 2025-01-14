// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "device.hpp"

#include "mesh_config.hpp"
#include "mesh_device_view.hpp"
#include "sub_device_types.hpp"
#include "span.hpp"

namespace tt::tt_metal {

class SubDeviceManagerTracker;

namespace distributed {

class MeshCommandQueue;
class MeshDeviceView;
class MeshSubDeviceManagerId;

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
            const DispatchCoreConfig& dispatch_core_config,
            const MeshDeviceConfig& config);

        // Destructor releases physical resources
        ~ScopedDevices();
        ScopedDevices(const ScopedDevices&) = delete;
        ScopedDevices& operator=(const ScopedDevices&) = delete;

        const std::vector<IDevice*>& get_devices() const;
    };

    std::shared_ptr<ScopedDevices> scoped_devices_;
    MeshDeviceID mesh_id_;
    MeshShape mesh_shape_;
    std::unique_ptr<MeshDeviceView> view_;
    std::vector<std::shared_ptr<MeshDevice>>
        submeshes_;                          // Parent owns submeshes and is responsible for their destruction
    std::weak_ptr<MeshDevice> parent_mesh_;  // Submesh created with reference to parent mesh
    std::unique_ptr<MeshCommandQueue> mesh_command_queue_;
    std::unique_ptr<SubDeviceManagerTracker> sub_device_manager_tracker_;

    // This is a reference device used to query properties that are the same for all devices in the mesh.
    IDevice* reference_device() const;

public:
    MeshDevice(
        std::shared_ptr<ScopedDevices> mesh_handle,
        const MeshShape& mesh_shape,
        std::weak_ptr<MeshDevice> parent_mesh = {});
    ~MeshDevice() override;

    MeshDevice(const MeshDevice&) = delete;
    MeshDevice& operator=(const MeshDevice&) = delete;

    MeshDevice(MeshDevice&&) = delete;
    MeshDevice& operator=(MeshDevice&&) = delete;

    void set_speculation_state(bool state, uint32_t p_tensor_addr) override;
    std::pair<bool, uint32_t> get_speculation_state() const override;
    void set_speculation_states(std::vector<bool> states, uint32_t p_tensor_addr) override;

    // IDevice interface implementation
    tt::ARCH arch() const override;
    MeshDeviceID id() const override;
    uint32_t build_key() const override;
    uint8_t num_hw_cqs() const override;
    bool is_initialized() const override;

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;

    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;
    CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const override;
    CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const override;
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>&logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment() override;


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
    CoreCoord compute_with_storage_grid_size() const override;
    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;
    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const override;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const override;
    const std::set<CoreCoord>& ethernet_cores() const override;
    const std::set<CoreCoord>& storage_only_cores() const override;
    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;
    const JitBuildEnv& build_env() const override;
    const string build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const string build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const override;
    const JitBuildState& build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const JitBuildState& build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const JitBuildStateSubset build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const override;
    SystemMemoryManager& sysmem_manager() override;
    CommandQueue& command_queue(size_t cq_id = 0) override;

    // Trace APIs
    void begin_trace(const uint8_t cq_id, const uint32_t tid) override;
    void end_trace(const uint8_t cq_id, const uint32_t tid) override;
    void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) override;
    void release_trace(const uint32_t tid) override;
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) override;
    uint32_t get_trace_buffers_size() const override;
    void set_trace_buffers_size(uint32_t size) override;
    // Light Metal
    void load_trace(uint8_t cq_id, uint32_t trace_id, const TraceDescriptor& trace_desc) override;

    bool using_slow_dispatch() const override;
    bool using_fast_dispatch() const override;

    // Initialization APIs
    bool initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {}, bool minimal = false) override;
    void build_firmware() override;
    void reset_cores() override;
    void initialize_and_launch_firmware() override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;
    bool close() override;
    void enable_async(bool enable) override;
    void synchronize() override;
    WorkExecutorMode get_worker_mode() override;
    bool is_worker_queue_empty() const override;
    void push_work(std::function<void()> work, bool blocking) override;
    void enable_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override;
    std::size_t num_program_cache_entries() override;
    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;
    std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(
        const std::vector<CoreRange>& ranges, const CoreType core_type) override;
    size_t get_device_kernel_defines_hash() override;
    uint8_t num_noc_mcast_txns(SubDeviceId sub_device_id) const override;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const override;
    uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data=true, bool unicast_data=true) const override;
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
    // TODO #16526: Temporary api until migration to actual fabric is complete
    std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    bool is_mmio_capable() const override;
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio() const override;

    // A MeshDevice is a collection of devices arranged in a 2D grid.
    // The type parameter allows the caller to specify how to linearize the devices in the mesh.

    // Returns the devices in the mesh in row-major order.
    std::vector<IDevice*> get_devices() const;
    IDevice* get_device_index(size_t logical_device_id) const;
    IDevice* get_device(chip_id_t physical_device_id) const;
    IDevice* get_device(size_t row_idx, size_t col_idx) const;

    const DeviceIds get_device_ids() const;

    size_t num_devices() const;
    size_t num_rows() const;
    size_t num_cols() const;
    MeshShape shape() const;

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
        const MeshShape& submesh_shape, const MeshOffset& offset = MeshOffset{0, 0});

    std::vector<std::shared_ptr<MeshDevice>> create_submeshes(const MeshShape& submesh_shape);

    // These methods will get removed once in favour of the ones in IDevice* and TT-Mesh bringup
    // These are prefixed with "mesh_" to avoid conflicts with the IDevice* methods
    MeshCommandQueue& mesh_command_queue();
    MeshSubDeviceManagerId mesh_create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    // TODO #16526: Temporary api until migration to actual fabric is complete
    std::tuple<MeshSubDeviceManagerId, SubDeviceId> mesh_create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    void mesh_clear_loaded_sub_device_manager();

    void mesh_load_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id);
    void mesh_remove_sub_device_manager(MeshSubDeviceManagerId mesh_sub_device_manager_id);
    // TODO #16492: Add get_sub_device_stall_group once MeshDevice is no longer just a collection of single Devices
    // and the MeshDevice has a single SubDeviceManager responsible for all Devices.
    void mesh_set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids);
    void mesh_reset_sub_device_stall_group();

    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {});
};

std::ostream& operator<<(std::ostream& os, const MeshDevice& mesh_device);

// TODO: This will be removed once we have DistributedDevice
// Currently required since each device manages its own sub-device manager ids
struct MeshSubDeviceManagerId {
    MeshSubDeviceManagerId(const MeshDevice& mesh_device);

    std::vector<SubDeviceManagerId> sub_device_manager_ids;
};

}  // namespace distributed

}  // namespace tt::tt_metal
