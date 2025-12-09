// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file exists in API directory for transitional reasons and WILL BE removed into `impl` in the future,
 * external users MUST NOT include or use anything here.
 */

#pragma once

#include <tt-metalium/device.hpp>

namespace tt::tt_metal {

class IDeviceImpl {
public:
    virtual ~IDeviceImpl() = default;
    virtual ChipId build_id() const = 0;
    virtual uint8_t num_hw_cqs() const = 0;
    virtual bool is_initialized() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;
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

    // Convert a logical coordinate to virtual coordinate for an ethernet coordinate
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const = 0;

    // Convert a virtual ethernet coordinate to logical coordinate
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const = 0;

    // Returns virtual coordinates of inactive ethernet cores
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;

    // Returns true if the ethernet core is active
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const = 0;

    // Returns virtual coordinates of active ethernet cores. Some ethernet cores may be reserved for dispatch use.
    //
    // This is also exposed in IDevice, but only referenced in programming examples and tests
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;

    virtual std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;

    virtual const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const = 0;

    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;

    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const = 0;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        tt::stl::Span<const SubDeviceId> sub_device_ids) const = 0;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    virtual const std::set<CoreCoord>& ethernet_cores() const = 0;

    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    virtual SystemMemoryManager& sysmem_manager() = 0;

    // If cq_id is not provided, the current command queue is returned from the current thread
    virtual CommandQueue& command_queue(std::optional<uint8_t> cq_id = std::nullopt) = 0;

    virtual uint32_t get_trace_buffers_size() const = 0;
    virtual void set_trace_buffers_size(uint32_t size) = 0;

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
    // Not used at all
    virtual void init_fabric() = 0;

    // Puts device into reset
    virtual bool close() = 0;

    // Not used
    virtual void clear_program_cache() = 0;
    // Not used
    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    // Not used
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
    // Not used
    virtual SubDeviceManagerId create_sub_device_manager(
        tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    // Not used
    virtual SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    // Not used
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    // Not used
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    // Not used
    virtual void clear_loaded_sub_device_manager() = 0;
    virtual CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const = 0;
    virtual const std::vector<SubDeviceId>& get_sub_device_stall_group() const = 0;
    // Not used
    virtual void set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) = 0;
    virtual void reset_sub_device_stall_group() = 0;
    virtual uint32_t num_sub_devices() const = 0;
    virtual uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) = 0;

    // Allowing to get corresponding MeshDevice for a given device to properly schedule programs / create buffers for
    // it. This is currently used exclusively by profiler.
    virtual std::shared_ptr<distributed::MeshDevice> get_mesh_device() = 0;
};

}  // namespace tt::tt_metal
