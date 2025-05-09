// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include <tt-metalium/device.hpp>
#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/kernel_structs.h>  // Leaked up to ttnn level from here
#include <tt-metalium/work_executor_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/trace_buffer.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/program_cache.hpp>

namespace tt::tt_metal {
class SubDeviceManagerTracker;

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
        uint32_t completion_queue_reader_core = 0,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    ~Device() override;

    // TODO: Add copy/move semantics
    Device(const Device &other) = delete;
    Device& operator=(const Device &other) = delete;

    Device(Device&& other);
    Device& operator=(Device&& other);

    tt::ARCH arch() const override;

    chip_id_t id() const override { return id_; }
    // For a single device, build id is the same as device id
    chip_id_t build_id() const override { return id_; }

    uint8_t num_hw_cqs() const override { return num_hw_cqs_; }

    bool is_initialized() const override { return this->initialized_; }

    int num_dram_channels() const override;
    uint32_t l1_size_per_core() const override;
    uint32_t dram_size_per_channel() const override;
    CoreCoord grid_size() const override;
    CoreCoord logical_grid_size() const override;
    CoreCoord dram_grid_size() const override;

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
    // `skip_reserved_tunnel_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const override;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const override;
    // `skip_reserved_tunnel_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
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

    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const override;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord> &ethernet_cores() const override { return this->ethernet_cores_; }

    const std::set<CoreCoord> &storage_only_cores() const override { return this->storage_only_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;

    SystemMemoryManager& sysmem_manager() override { return *sysmem_manager_; }
    CommandQueue& command_queue(size_t cq_id = 0) override;

    // Metal trace device capture mode
    void begin_trace(const uint8_t cq_id, const uint32_t tid) override;
    void end_trace(const uint8_t cq_id, const uint32_t tid) override;
    void replay_trace(
        const uint8_t cq_id,
        const uint32_t tid,
        const bool block_on_device,
        const bool block_on_worker_thread) override;
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
    bool initialize(
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) override;
    void reset_cores() override;
    void initialize_and_launch_firmware() override;
    void init_command_queue_host() override;
    void init_command_queue_device() override;

    void init_fabric() override;
    void set_ethernet_core_count_on_dispatcher(uint32_t num_ethernet_cores);
    uint32_t get_ethernet_core_count_on_dispatcher() const;
    // Puts device into reset
    bool close() override;

    // Program cache interface. Synchronize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override { return program_cache_; }
    std::size_t num_program_cache_entries() override;

    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;
    HalMemType get_mem_type_of_core(CoreCoord virtual_core) const override;

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
    bool dispatch_firmware_active() const override { return dispatch_firmware_active_; };
    // TODO #15944: Temporary api until migration to actual fabric is complete
    std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;

    bool is_mmio_capable() const override;
    // TODO #20966: Remove these APIs
    std::shared_ptr<distributed::MeshDevice> get_mesh_device() override;
    void set_mesh_device(std::shared_ptr<distributed::MeshDevice> mesh_device) { this->mesh_device = mesh_device; };

private:
    static constexpr uint32_t DEFAULT_NUM_SUB_DEVICES = 1;

    void initialize_cluster();
    std::unique_ptr<Allocator> initialize_allocator(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {});
    void initialize_device_bank_to_noc_tables(const HalProgrammableCoreType &core_type, CoreCoord virtual_core);
    void initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord virtual_core, launch_msg_t *launch_msg, go_msg_t* go_msg);

    void initialize_default_sub_device_state(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        tt::stl::Span<const std::uint32_t> l1_bank_remap);

    void compile_command_queue_programs();
    void configure_command_queue_programs();
    void clear_l1_state();
    void clear_dram_state();
    void clear_launch_messages_on_eth_cores();
    void get_associated_dispatch_virtual_cores(
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>>& my_dispatch_cores,
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>>& other_dispatch_cores);

    void generate_device_bank_to_noc_tables();

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    CoreCoord physical_worker_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel) const;
    CoreCoord virtual_core_from_physical_core(const CoreCoord& physical_coord) const;

    chip_id_t id_;
    std::vector<std::vector<chip_id_t>> tunnels_from_mmio_;

    std::unique_ptr<SubDeviceManagerTracker> sub_device_manager_tracker_;

    bool initialized_ = false;
    // This variable tracks the state of dispatch firmware on device.
    // It is set to true when dispatch firmware is launched, and reset
    // after the terimnate command is sent.
    bool dispatch_firmware_active_ = false;

    std::vector<std::unique_ptr<Program>> command_queue_programs_;
    bool using_fast_dispatch_ = false;
    // TODO #20966: Remove this member
    std::weak_ptr<distributed::MeshDevice> mesh_device;

    // Fabric program includes ethernet router kernel
    std::unique_ptr<Program> fabric_program_;

    uint32_t worker_thread_core_ = 0;
    uint32_t completion_queue_reader_core_ = 0;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    uint8_t num_hw_cqs_ = 1;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<CommandQueue>> command_queues_;

    std::set<CoreCoord> compute_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;
    std::vector<CoreCoord> optimal_dram_bank_to_logical_worker_assignment_;

    std::vector<int32_t> dram_bank_offset_map_;
    std::vector<int32_t> l1_bank_offset_map_;
    std::vector<uint16_t> dram_bank_to_noc_xy_;
    std::vector<uint16_t> l1_bank_to_noc_xy_;
    std::shared_ptr<Buffer> dram_debug_buffer_;

    program_cache::detail::ProgramCache program_cache_;

    uint32_t trace_buffers_size_ = 0;
    bool uninitialized_error_fired_ =
        false;  // To avoid spam with warnings about calling Device methods when it's not initialized.
    uint32_t ethernet_core_count_on_dispatcher_ = 0;
};

}  // namespace tt::tt_metal
