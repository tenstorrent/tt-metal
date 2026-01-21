// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/device.hpp>
#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/kernel_structs.h>  // Leaked up to ttnn level from here
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include "impl/dispatch/command_queue.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/sub_device.hpp>
#include "trace/trace_buffer.hpp"
#include <tt_stl/span.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/experimental/device.hpp>

namespace tt::tt_metal {
class SubDeviceManagerTracker;
class AllocatorImpl;

// A physical PCIexpress Tenstorrent device
class Device : public IDevice {
public:
    Device() = delete;
    Device(
        ChipId device_id,
        uint8_t num_hw_cqs,
        std::size_t l1_small_size,
        std::size_t trace_region_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false,
        uint32_t worker_thread_core = 0,
        uint32_t completion_queue_reader_core = 0,
        std::size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    ~Device() override;

    // TODO: Add copy/move semantics
    Device(const Device& other) = delete;
    Device& operator=(const Device& other) = delete;

    Device(Device&& other) noexcept;
    Device& operator=(Device&& other) noexcept;

    tt::ARCH arch() const override;

    ChipId id() const override { return id_; }
    // For a single device, build id is the same as device id
    ChipId build_id() const override { return id_; }

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

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const override;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) override;

    CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const override;
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const override;

    // Ethernet API
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const override;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const override;
    // `skip_reserved_tunnel_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const override;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const override;
    // `skip_reserved_tunnel_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const override;
    std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const override;
    std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const override;
    bool is_inactive_ethernet_core(CoreCoord logical_core) const override;
    uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) override;

    CoreCoord compute_with_storage_grid_size() const override;

    const std::unique_ptr<Allocator>& allocator() const override;
    const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const override;

    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const override;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const override;
    uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const override;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord>& ethernet_cores() const override { return this->ethernet_cores_; }

    const std::set<CoreCoord>& storage_only_cores() const override { return this->storage_only_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const override;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const override;

    SystemMemoryManager& sysmem_manager() override;
    CommandQueue& command_queue(std::optional<uint8_t> cq_id = std::nullopt) override;

    // Metal trace device capture mode
    uint32_t get_trace_buffers_size() const override { return trace_buffers_size_; }
    void set_trace_buffers_size(uint32_t size) override { trace_buffers_size_ = size; }

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
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
    // Puts device into reset
    bool close() override;

    // Program cache interface. Synchronize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache() override;
    void clear_program_cache() override;
    void disable_and_clear_program_cache() override;
    program_cache::detail::ProgramCache& get_program_cache() override { return program_cache_; }
    std::size_t num_program_cache_entries() override;

    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const override;
    HalMemType get_mem_type_of_core(CoreCoord virtual_core) const override;

    uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const override;

    CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const override;

    bool is_mmio_capable() const override;
    // TODO #20966: Remove these APIs
    std::shared_ptr<distributed::MeshDevice> get_mesh_device() override;
    void set_mesh_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
        this->mesh_device = mesh_device;
    };

private:
    // Depracated ovverrides for sub_device_manager_tracker
    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl() const override;
    const std::unique_ptr<AllocatorImpl>& allocator_impl(SubDeviceId sub_device_id) const override;
    uint32_t num_sub_devices() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const override;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const override;
    bool has_noc_mcast_txns(SubDeviceId sub_device_id) const override;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const override;
    SubDeviceManagerId get_active_sub_device_manager_id() const override;
    SubDeviceManagerId get_default_sub_device_manager_id() const override;
    SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) override;
    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) override;
    void clear_loaded_sub_device_manager() override;
    const std::vector<SubDeviceId>& get_sub_device_ids() const override;
    const std::vector<SubDeviceId>& get_sub_device_stall_group() const override;
    void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) override;
    void reset_sub_device_stall_group() override;

    static constexpr uint32_t DEFAULT_NUM_SUB_DEVICES = 1;

    std::unique_ptr<AllocatorImpl> initialize_allocator(
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {});

    void configure_command_queue_programs();

    // NOLINTNEXTLINE(readability-make-member-function-const)
    void mark_allocations_unsafe();
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void mark_allocations_safe();

    CoreCoord physical_worker_core_from_logical_core(const CoreCoord& logical_core) const;
    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel, NOC noc = NOC::NOC_0) const;
    CoreCoord virtual_core_from_physical_core(const CoreCoord& physical_coord) const;

    ChipId id_;
    std::vector<std::vector<ChipId>> tunnels_from_mmio_;

    bool initialized_ = false;

    std::vector<std::unique_ptr<Program>> command_queue_programs_;
    bool using_fast_dispatch_ = false;

    // TODO #20966: Remove this member
    std::weak_ptr<distributed::MeshDevice> mesh_device;

    // Fabric program includes ethernet router kernel
    std::unique_ptr<Program> fabric_program_;

    uint32_t completion_queue_reader_core_ = 0;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    uint8_t num_hw_cqs_ = 1;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<CommandQueue>> command_queues_;

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

    std::unique_ptr<AllocatorImpl> default_allocator_;

    // Friend declaration for experimental API
    friend uint32_t experimental::Device::get_worker_noc_hop_distance(
        IDevice* device, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc);
};

}  // namespace tt::tt_metal
