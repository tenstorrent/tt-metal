// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include "impl/dispatch/work_executor.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/program/program_device_map.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/tt_stl/span.hpp"
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {
// Fwd declares
enum class BufferType;

inline namespace v0 {

class Buffer;
class Program;
class CommandQueue;
class SubDevice;

}  // namespace v0

class JitBuildEnv;
class HWCommandQueue;
class TraceBuffer;

using on_close_device_callback = std::function<void()>;

inline namespace v0 {

class IDevice {
public:
    Device() = delete;
    virtual ~Device() = default;

    Device(const Device& other) = delete;
    Device& operator=(const Device& other) = delete;

    Device(Device&& other) = default;
    Device& operator=(Device&& other) = default;

    virtual tt::ARCH arch() const;

    virtual chip_id_t id() const;

    virtual uint32_t build_key() const;

    virtual uint8_t num_hw_cqs() const;

    virtual bool is_initialized() const;

    virtual int num_dram_channels() const;
    virtual uint32_t l1_size_per_core() const;
    virtual uint32_t dram_size_per_channel() const;
    virtual CoreCoord grid_size() const;
    virtual CoreCoord logical_grid_size() const;
    virtual CoreCoord dram_grid_size() const;
    virtual CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const;

    // Given a Virtual coordinate in noc_index space, get the equivalent coordinate in Virtual NOC0 space
    virtual CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const;
    // Given a coordinate in Virtual NOC0 Space, get the equivalent coordinate in Virtual noc_index space
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const;

    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const;
    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord>& logical_cores) const;
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment();

    virtual CoreCoord virtual_core_from_logical_core(const CoreCoord& logical_coord, const CoreType& core_type) const;
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const;

    // Ethernet API
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const;
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const;
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const;
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const;
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const;
    virtual std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const;
    virtual std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const;

    virtual CoreCoord compute_with_storage_grid_size() const;

    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const;
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const;

    virtual const std::unique_ptr<Allocator>& get_initialized_allocator() const;
    virtual const std::unique_ptr<Allocator>& get_initialized_allocator(SubDeviceId sub_device_id) const;

    virtual DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const;
    virtual DeviceAddr get_base_allocator_addr(const HalMemType& mem_type, SubDeviceId sub_device_id) const;

    virtual uint32_t num_banks(const BufferType& buffer_type) const;
    virtual uint32_t num_banks(const BufferType& buffer_type, SubDeviceId sub_device_id) const;
    virtual uint32_t bank_size(const BufferType& buffer_type) const;
    virtual uint32_t bank_size(const BufferType& buffer_type, SubDeviceId sub_device_id) const;

    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;
    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const;

    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const;
    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const;

    virtual int32_t bank_offset(BufferType buffer_type, uint32_t bank_id) const;
    virtual int32_t bank_offset(BufferType buffer_type, uint32_t bank_id, SubDeviceId sub_device_id) const;

    virtual CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;
    virtual CoreCoord logical_core_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const;

    virtual const std::vector<uint32_t>& bank_ids_from_dram_channel(uint32_t dram_channel) const;
    virtual const std::vector<uint32_t>& bank_ids_from_dram_channel(
        uint32_t dram_channel, SubDeviceId sub_device_id) const;

    virtual const std::vector<uint32_t>& bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;
    virtual const std::vector<uint32_t>& bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core, SubDeviceId sub_device_id) const;

    virtual allocator::Statistics get_memory_allocation_statistics(const BufferType& buffer_type) const;
    virtual allocator::Statistics get_memory_allocation_statistics(
        const BufferType& buffer_type, SubDeviceId sub_device_id) const;

    virtual uint32_t get_allocator_alignment() const;
    virtual uint32_t get_allocator_alignment(SubDeviceId sub_device_id) const;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const;
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const;

    virtual size_t get_l1_small_size() const;
    virtual size_t get_l1_small_size(SubDeviceId sub_device_id) const;

    virtual const std::unordered_set<Buffer*>& get_allocated_buffers() const;
    virtual const std::unordered_set<Buffer*>& get_allocated_buffers(SubDeviceId sub_device_id) const;

    virtual void deallocate_buffers();
    virtual void deallocate_buffers(SubDeviceId sub_device_id);

    virtual void dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out) const;
    virtual void dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out, SubDeviceId sub_device_id) const;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    virtual const std::set<CoreCoord>& ethernet_cores() const;

    virtual const std::set<CoreCoord>& storage_only_cores() const;

    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const;

    // machine epsilon
    virtual float sfpu_eps() const;

    // machine nan
    virtual float sfpu_nan() const;

    // machine inf
    virtual float sfpu_inf() const;

    virtual const JitBuildEnv& build_env() const { return this->build_env_; }
    virtual const string build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const;
    virtual const string build_kernel_target_path(
        uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const;
    virtual const JitBuildState& build_firmware_state(
        uint32_t programmable_core, uint32_t processor_class, int i) const;
    virtual const JitBuildState& build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const;
    virtual const JitBuildStateSubset build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const;

    virtual SystemMemoryManager& sysmem_manager();
    virtual HWCommandQueue& hw_command_queue(size_t cq_id = 0);
    virtual CommandQueue& command_queue(size_t cq_id = 0);

    // Metal trace device capture mode
    virtual void begin_trace(const uint8_t cq_id, const uint32_t tid);
    virtual void end_trace(const uint8_t cq_id, const uint32_t tid);
    virtual void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking);
    virtual void release_trace(const uint32_t tid);
    virtual std::shared_ptr<TraceBuffer> get_trace(uint32_t tid);
    virtual uint32_t get_trace_buffers_size() const;
    virtual void set_trace_buffers_size(uint32_t size);

    virtual bool using_slow_dispatch() const;
    virtual bool using_fast_dispatch() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    virtual bool initialize(
        const uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false);
    virtual void build_firmware();
    virtual void reset_cores();
    virtual void initialize_and_launch_firmware();
    virtual void init_command_queue_host();
    virtual void init_command_queue_device();
    virtual void initialize_synchronous_sw_cmd_queue();
    virtual void update_dispatch_cores_for_multi_cq_eth_dispatch();

    // Puts device into reset
    virtual bool close();
    friend bool CloseDevice(Device* device);

    virtual void enable_async(bool enable);
    virtual void synchronize();
    virtual WorkExecutorMode get_worker_mode();
    virtual void set_worker_queue_mode(const WorkerQueueMode& mode);
    virtual WorkerQueueMode get_worker_queue_mode();
    virtual bool is_worker_queue_empty() const;
    virtual bool can_use_passthrough_scheduling() const;

    template <typename F>
    void push_work(F&& work, bool blocking = false) {
        this->work_executor_.push_work(std::forward<F>(work), blocking);
    }

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    virtual void enable_program_cache();
    virtual void disable_and_clear_program_cache();
    virtual program_cache::detail::ProgramCache& get_program_cache();
    virtual std::size_t num_program_cache_entries();

    virtual HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const;

    template <typename T = DeviceAddr>
    virtual T get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;

    virtual std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(
        const std::vector<CoreRange>& ranges, const CoreType core_type);

    virtual bool dispatch_s_enabled() const;
    virtual bool distributed_dispatcher() const;
    virtual NOC dispatch_go_signal_noc() const;
    virtual size_t get_device_kernel_defines_hash();

    virtual uint8_t num_noc_mcast_txns(SubDeviceId sub_device_id) const;
    virtual uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const;
    virtual uint8_t noc_data_start_index(
        SubDeviceId sub_device_id, bool mcast_data = true, bool unicast_data = true) const;

    virtual SubDeviceManagerId get_active_sub_device_manager_id() const;
    virtual SubDeviceManagerId get_default_sub_device_manager_id() const;
    virtual SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
    virtual void clear_loaded_sub_device_manager();
    virtual LaunchMessageRingBufferState& get_worker_launch_message_buffer_state(SubDeviceId sub_device_id);
    virtual const std::vector<SubDeviceId>& get_sub_device_ids() const;
    virtual uint32_t num_sub_devices() const;

    // TODO #15944: Temporary api until migration to actual fabric is complete
    virtual std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    virtual std::optional<SubDeviceId> get_fabric_sub_device_id() const;

    virtual uint32_t get_completion_queue_reader_core() const;

    virtual bool is_mmio_capable() const;
    virtual std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio() const;

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
};

}  // namespace v0

template <typename T>
inline T IDevice::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return hal.get_dev_addr<T>(this->get_programmable_core_type(virtual_core), addr_type);
}

}  // namespace tt_metal

}  // namespace tt
