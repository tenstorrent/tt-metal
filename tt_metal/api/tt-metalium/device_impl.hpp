// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include "device.hpp"
#include "hostdevcommon/common_values.hpp"
#include "work_executor.hpp"
#include "basic_allocator.hpp"
#include "l1_banking_allocator.hpp"
#include "data_types.hpp"
#include "program_device_map.hpp"
#include "build.hpp"
#include "tt_cluster.hpp"
#include "hal.hpp"
#include "command_queue_interface.hpp"
#include "command_queue.hpp"
#include "sub_device_manager_tracker.hpp"
#include "sub_device_types.hpp"
#include "trace_buffer.hpp"
#include "span.hpp"
#include "program_cache.hpp"

namespace tt::tt_metal {

inline namespace v0 {

// A physical PCIexpress Tenstorrent device
class Device : public IDevice {
public:
    // friend void tt_gdb(IDevice* device, int chip_id, const vector<CoreCoord> cores, vector<string> ops);
    Device () = delete;
    Device(
        chip_id_t device_id,
        const uint8_t num_hw_cqs,
        std::size_t l1_small_size,
        std::size_t trace_region_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false,
        uint32_t worker_thread_core = 0,
        uint32_t completion_queue_reader_core = 0);

    ~Device() override;

    // TODO: Add copy/move semantics
    Device(const Device &other) = delete;
    Device& operator=(const Device &other) = delete;

    Device(Device &&other) = default;
    Device& operator=(Device &&other) = default;

    tt::ARCH arch() const override;

    chip_id_t id() const override { return id_; }

    uint32_t build_key() const override { return build_key_; }

    uint8_t num_hw_cqs() const override { return num_hw_cqs_; }

    bool is_initialized() const override { return this->initialized_; }

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;
    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;
    CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const override;
    void set_speculation_states(std::vector<bool> states, uint32_t p_tensor_addr) override;
    void set_speculation_state(bool state, uint32_t p_tensor_addr) override;
    std::pair<bool, uint32_t> get_speculation_state() const override;
    // Given a Virtual coordinate in noc_index space, get the equivalent coordinate in Virtual NOC0 space
    CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const override;
    // Given a coordinate in Virtual NOC0 Space, get the equivalent coordinate in Virtual noc_index space
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const override;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment() override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const override;

    // Ethernet API
    CoreCoord ethernet_core_from_logical_core(const CoreCoord &logical_core) const override;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const override;
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
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const override;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord> &ethernet_cores() const override { return this->ethernet_cores_; }

    const std::set<CoreCoord> &storage_only_cores() const override { return this->storage_only_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;

    const JitBuildEnv& build_env() const override { return this->build_env_; }
    const string build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const string build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const override;
    const JitBuildState& build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const JitBuildState& build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const override;
    const JitBuildStateSubset build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const override;

    SystemMemoryManager& sysmem_manager() override { return *sysmem_manager_; }
    CommandQueue& command_queue(size_t cq_id = 0) override;

    // Metal trace device capture mode
    void begin_trace(const uint8_t cq_id, const uint32_t tid) override;
    void end_trace(const uint8_t cq_id, const uint32_t tid) override;
    void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) override;
    void release_trace(const uint32_t tid) override;
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) override;
    uint32_t get_trace_buffers_size() const override { return trace_buffers_size_; }
    void set_trace_buffers_size(uint32_t size) override { trace_buffers_size_ = size; }
    // Light Metal
    void load_trace(uint8_t cq_id, uint32_t trace_id, const TraceDescriptor& trace_desc) override;

    bool using_slow_dispatch() const override;
    bool using_fast_dispatch() const override;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {}, bool minimal = false) override;
    void build_firmware() override;
    void reset_cores() override;
    void initialize_and_launch_firmware() override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;

    // Puts device into reset
    bool close() override;

    void enable_async(bool enable) override;
    void synchronize() override;
    WorkExecutorMode get_worker_mode() override { return work_executor_.get_worker_mode(); }
    bool is_worker_queue_empty() const override { return work_executor_.worker_queue.empty(); }

    void push_work(std::function<void()> work, bool blocking) override;

    // Program cache interface. Synchronize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override { return program_cache_; }
    std::size_t num_program_cache_entries() override;

    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;

    std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(const std::vector<CoreRange>& ranges, const CoreType core_type) override;

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
    const std::vector<SubDeviceId> &get_sub_device_ids() const override;
    const std::vector<SubDeviceId> &get_sub_device_stall_group() const override;
    void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) override;
    void reset_sub_device_stall_group() override;
    uint32_t num_sub_devices() const override;

    // TODO #15944: Temporary api until migration to actual fabric is complete
    std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;

    bool is_mmio_capable() const override;
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio() const override { return tunnels_from_mmio_; }

private:
    bool waiting_for_speculation_ = false;
    uint32_t p_tensor_addr_ = 0;
    static constexpr uint32_t DEFAULT_NUM_SUB_DEVICES = 1;

    void initialize_cluster();
    std::unique_ptr<Allocator> initialize_allocator(
        size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {});
    void initialize_build();
    void initialize_device_kernel_defines();
    void initialize_device_bank_to_noc_tables(const HalProgrammableCoreType &core_type, CoreCoord virtual_core);
    void initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord virtual_core, launch_msg_t *launch_msg, go_msg_t* go_msg);

    void initialize_default_sub_device_state(size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap);

    void update_dispatch_cores_for_multi_cq_eth_dispatch();

    void compile_command_queue_programs();
    void configure_command_queue_programs();
    void clear_l1_state();
    void get_associated_dispatch_virtual_cores(
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &my_dispatch_cores,
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &other_dispatch_cores);
    std::pair<int, int> build_processor_type_to_index(uint32_t programmable_core, uint32_t processor_class) const;

    void set_worker_mode(const WorkExecutorMode& mode);

    void generate_device_bank_to_noc_tables();

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    CoreCoord physical_worker_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel) const;
    CoreType core_type_from_physical_core(const CoreCoord &physical_core) const;
    CoreCoord virtual_core_from_physical_core(const CoreCoord& physical_coord) const;

    chip_id_t id_;
    uint32_t build_key_ = 0;
    std::vector<std::vector<chip_id_t>> tunnels_from_mmio_;

    std::unique_ptr<SubDeviceManagerTracker> sub_device_manager_tracker_;

    bool initialized_ = false;

    std::vector<std::unique_ptr<Program>> command_queue_programs_;
    bool using_fast_dispatch_ = false;

    // Work Executor for this device - can asynchronously process host side work for
    // all tasks scheduled on this device
    WorkExecutor work_executor_;
    uint32_t worker_thread_core_ = 0;
    uint32_t completion_queue_reader_core_ = 0;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    uint8_t num_hw_cqs_ = 1;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<CommandQueue>> command_queues_;

    JitBuildEnv build_env_;
    JitBuildStateSet firmware_build_states_;
    JitBuildStateSet kernel_build_states_;
    std::vector<std::vector<std::pair<int, int>>> build_state_indices_;

    std::set<CoreCoord> compute_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;
    std::vector<CoreCoord> optimal_dram_bank_to_logical_worker_assignment_;

    std::map<std::string, std::string> device_kernel_defines_;

    std::vector<int32_t> dram_bank_offset_map_;
    std::vector<int32_t> l1_bank_offset_map_;
    std::vector<uint16_t> dram_bank_to_noc_xy_;
    std::vector<uint16_t> l1_bank_to_noc_xy_;

    program_cache::detail::ProgramCache program_cache_;

    uint32_t trace_buffers_size_ = 0;
};

}  // namespace v0
}  // namespace tt::tt_metal
