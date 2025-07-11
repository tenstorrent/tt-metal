// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/kernel_structs.h>  // Not used here, but leaked to programming examples
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/allocator_types.hpp>

#include <tt_stl/span.hpp>

namespace tt {

namespace tt_metal {

namespace program_cache::detail {
class ProgramCache;
}
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;
enum class BufferType;

class Buffer;
class Program;
class SubDevice;

class CommandQueue;
class SystemMemoryManager;
class TraceBuffer;
struct TraceDescriptor;

namespace distributed {
class MeshDevice;
}

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
    virtual chip_id_t build_id() const = 0;

    virtual uint8_t num_hw_cqs() const = 0;

    virtual bool is_initialized() const = 0;

    virtual int num_dram_channels() const = 0;
    virtual uint32_t l1_size_per_core() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;
    virtual CoreCoord grid_size() const = 0;
    virtual CoreCoord logical_grid_size() const = 0;
    virtual CoreCoord dram_grid_size() const = 0;

    // Given a coordinate in Virtual NOC0 Space, get the equivalent coordinate in Virtual noc_index space
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;

    // Convert logical coordinates to virtual coordinates for worker coordinates
    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;

    // Convert logical coordinates to virtaul coordinates for ethernet coordinates
    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;

    // Returns the optimal DRAM bank coordinates to logical worker assignment based on which noc will be issuing DRAM
    // requests
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) = 0;

    // Convert a logical coordinate to virtual coordinate
    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord, const CoreType& core_type) const = 0;

    // Convert a logical coordinate to a virtual coordinate for a worker coordinate
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const = 0;

    // Convert a logical coordinate to virtual coordinate for an ethernet coordinate
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const = 0;

    // Convert a virtual ethernet coordinate to logical coordinate
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const = 0;

    // Returns virtual coordinates of active ethernet cores. Some ethernet cores may be reserved for dispatch use.
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;

    // Returns virtual coordinates of inactive ethernet cores
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;

    // Returns true if the ethernet core is active
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const = 0;
    virtual std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;
    virtual std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const = 0;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const = 0;

    virtual CoreCoord compute_with_storage_grid_size() const = 0;

    // Returns a logical CoreRangeSet of the worker cores in the specified sub device that was previously loaded.
    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;
    // Returns the number of worker cores in the specified sub device that was previously loaded.
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    virtual const std::unique_ptr<Allocator>& allocator() const = 0;
    virtual const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const = 0;

    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;
    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const = 0;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const = 0;
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const = 0;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    virtual const std::set<CoreCoord> &ethernet_cores() const = 0;
    virtual const std::set<CoreCoord> &storage_only_cores() const = 0;

    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    virtual SystemMemoryManager& sysmem_manager() = 0;
    virtual CommandQueue& command_queue(size_t cq_id = 0) = 0;

    // Metal trace device capture mode
    virtual void begin_trace(uint8_t cq_id, uint32_t tid) = 0;
    virtual void end_trace(uint8_t cq_id, uint32_t tid) = 0;
    virtual void replay_trace(uint8_t cq_id, uint32_t tid, bool block_on_device, bool block_on_worker_thread) = 0;
    virtual void release_trace(uint32_t tid) = 0;

    virtual std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) = 0;
    virtual uint32_t get_trace_buffers_size() const = 0;
    virtual void set_trace_buffers_size(uint32_t size) = 0;

    // Light Metal
    virtual void load_trace(uint8_t cq_id, uint32_t trace_id, const TraceDescriptor& trace_desc) = 0;
    virtual bool using_slow_dispatch() const = 0;
    virtual bool using_fast_dispatch() const = 0;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    virtual bool initialize(
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_size,
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) = 0;
    virtual void init_command_queue_host() = 0;
    virtual void init_command_queue_device() = 0;

    // return false if compile fails (mainly come from Nebula on TG)
    virtual bool compile_fabric() = 0;
    virtual void configure_fabric() = 0;
    virtual void init_fabric() = 0;
    // Puts device into reset
    virtual bool close() = 0;

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    virtual void enable_program_cache() = 0;
    virtual void clear_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    void set_program_cache_misses_allowed(bool allowed);
    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    virtual HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const = 0;
    virtual HalMemType get_mem_type_of_core(CoreCoord virtual_core) const = 0;

    // Returns the starting address and memory region size on the device for a given virtual core and L1 memory type
    uint64_t get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;
    uint64_t get_dev_size(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;

    virtual bool has_noc_mcast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const = 0;

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
    virtual uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) = 0;

    virtual bool is_mmio_capable() const = 0;

    // Allowing to get corresponding MeshDevice for a given device to properly schedule programs / create buffers for
    // it. This is currently used exclusively by profiler.
    virtual std::shared_ptr<distributed::MeshDevice> get_mesh_device() = 0;

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
};

}  // namespace tt_metal

}  // namespace tt
