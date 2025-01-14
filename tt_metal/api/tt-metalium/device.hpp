// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <utility>

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
#include "sub_device_manager.hpp"
#include "sub_device_types.hpp"
#include "span.hpp"
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;
enum class BufferType;

inline namespace v0 {

class Buffer;
class Program;
class SubDevice;

}  // namespace v0

class JitBuildEnv;
class CommandQueue;
class TraceBuffer;
struct TraceDescriptor;

namespace detail {
struct TraceDescriptor;
}

inline namespace v0 {

class IDevice {
public:
    IDevice() = default;
    virtual ~IDevice() = default;

    IDevice(const IDevice &other) = delete;
    IDevice& operator=(const IDevice &other) = delete;

    IDevice(IDevice &&other) = default;
    IDevice& operator=(IDevice &&other) = default;

    virtual tt::ARCH arch() const = 0;

    virtual chip_id_t id() const = 0;

    virtual uint32_t build_key() const = 0;

    virtual uint8_t num_hw_cqs() const = 0;

    virtual bool is_initialized() const = 0;

    virtual int num_dram_channels() const = 0;
    virtual uint32_t l1_size_per_core() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;
    virtual CoreCoord grid_size() const = 0;
    virtual CoreCoord logical_grid_size() const = 0;
    virtual CoreCoord dram_grid_size() const = 0;
    virtual CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const = 0;

    virtual void set_speculation_states(std::vector<bool> states, uint32_t p_tensor_addr) = 0;
    virtual void set_speculation_state(bool state, uint32_t p_tensor_addr) = 0;
    virtual std::pair<bool, uint32_t> get_speculation_state() const = 0;

    // Given a Virtual coordinate in noc_index space, get the equivalent coordinate in Virtual NOC0 space
    virtual CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;
    // Given a coordinate in Virtual NOC0 Space, get the equivalent coordinate in Virtual noc_index space
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;

    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const = 0;
    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const = 0;
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment() = 0;

    virtual CoreCoord virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const = 0;
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const = 0;

    // Ethernet API
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord &logical_core) const = 0;
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const = 0;
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const = 0;
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const = 0;
    virtual std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;
    virtual std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const = 0;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const = 0;

    virtual CoreCoord compute_with_storage_grid_size() const = 0;

    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    virtual const std::unique_ptr<Allocator>& allocator() const = 0;
    virtual const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const = 0;

    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;
    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const = 0;
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const = 0;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    virtual const std::set<CoreCoord> &ethernet_cores() const = 0;
    virtual const std::set<CoreCoord> &storage_only_cores() const = 0;

    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    virtual const JitBuildEnv& build_env() const = 0;
    virtual const string build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const = 0;
    virtual const string build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const = 0;
    virtual const JitBuildState& build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const = 0;
    virtual const JitBuildState& build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const = 0;
    virtual const JitBuildStateSubset build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const = 0;

    virtual SystemMemoryManager& sysmem_manager() = 0;
    virtual CommandQueue& command_queue(size_t cq_id = 0) = 0;

    // Metal trace device capture mode
    virtual void begin_trace(const uint8_t cq_id, const uint32_t tid) = 0;
    virtual void end_trace(const uint8_t cq_id, const uint32_t tid) = 0;
    virtual void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) = 0;
    virtual void release_trace(const uint32_t tid) = 0;

    virtual std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) = 0;
    virtual uint32_t get_trace_buffers_size() const = 0;
    virtual void set_trace_buffers_size(uint32_t size) = 0;

    // Light Metal
    virtual void load_trace(uint8_t cq_id, uint32_t trace_id, const TraceDescriptor& trace_desc) = 0;
    virtual bool using_slow_dispatch() const = 0;
    virtual bool using_fast_dispatch() const = 0;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    virtual bool initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {}, bool minimal = false) = 0;
    virtual void build_firmware() = 0;
    virtual void reset_cores() = 0;
    virtual void initialize_and_launch_firmware() = 0;
    virtual void init_command_queue_host() = 0;
    virtual void init_command_queue_device() = 0;

    // Puts device into reset
    virtual bool close() = 0;

    virtual void enable_async(bool enable) = 0;
    virtual void synchronize() = 0;
    virtual WorkExecutorMode get_worker_mode() = 0;
    virtual bool is_worker_queue_empty() const = 0;

    virtual void push_work(std::function<void()> work, bool blocking = false) = 0;

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    virtual void enable_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    virtual HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const = 0;

    template <typename T = DeviceAddr>
    T get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;

    virtual std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(const std::vector<CoreRange>& ranges, const CoreType core_type) = 0;

    virtual size_t get_device_kernel_defines_hash() = 0;

    virtual uint8_t num_noc_mcast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data=true, bool unicast_data=true) const = 0;

    virtual SubDeviceManagerId get_active_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId get_default_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void clear_loaded_sub_device_manager() = 0;
    virtual CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const = 0;
    virtual const std::vector<SubDeviceId> &get_sub_device_ids() const = 0;
    virtual const std::vector<SubDeviceId> &get_sub_device_stall_group() const = 0;
    virtual void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) = 0;
    virtual void reset_sub_device_stall_group() = 0;
    virtual uint32_t num_sub_devices() const = 0;

    // TODO #15944: Temporary api until migration to actual fabric is complete
    virtual std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;

    virtual bool is_mmio_capable() const = 0;
    virtual std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio() const = 0;

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
};

}  // namespace v0

template <typename T>
inline T IDevice::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return hal.get_dev_addr<T>(this->get_programmable_core_type(virtual_core), addr_type);
}

}  // namespace tt_metal

}  // namespace tt
