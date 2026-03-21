// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

class Allocator;
class AllocatorImpl;
class SubDevice;
class SystemMemoryManager;

namespace distributed {
class MeshDevice;
}

// Internal device interface containing methods that should not be part of the public API surface.
// These methods are used internally by tt_metal and tests.
// Access via IDevice::device_internal().
class IDeviceInternal {
public:
    virtual ~IDeviceInternal() = default;

    // Device Identity/State
    virtual ChipId build_id() const = 0;
    virtual uint8_t num_hw_cqs() const = 0;
    virtual bool is_initialized() const = 0;

    // Core Coordinate Translation
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;
    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;
    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;
    virtual uint32_t dram_channel_from_virtual_core(const CoreCoord& virtual_core) const = 0;
    virtual HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const = 0;
    virtual HalMemType get_mem_type_of_core(CoreCoord virtual_core) const = 0;

    // Ethernet Core Methods
    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const = 0;
    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;
    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const = 0;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const = 0;
    virtual std::tuple<ChipId, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;
    virtual std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const = 0;
    virtual const std::set<CoreCoord>& ethernet_cores() const = 0;

    // Memory/Allocator Internal
    virtual const std::unique_ptr<AllocatorImpl>& allocator_impl() const = 0;
    virtual const std::unique_ptr<AllocatorImpl>& allocator_impl(SubDeviceId sub_device_id) const = 0;
    virtual SystemMemoryManager& sysmem_manager() = 0;
    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    // Trace Buffers
    virtual uint32_t get_trace_buffers_size() const = 0;
    virtual void set_trace_buffers_size(uint32_t size) = 0;

    // Initialization/Lifecycle
    virtual bool initialize(
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_size,
        ttsl::Span<const std::uint32_t> l1_bank_remap = {},
        bool minimal = false) = 0;

    // Program Cache Internal
    virtual void clear_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    // Sub-Device Management Internal
    virtual SubDeviceManagerId create_sub_device_manager(
        ttsl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void clear_loaded_sub_device_manager() = 0;
    virtual CoreCoord virtual_program_dispatch_core(uint8_t cq_id) const = 0;
    virtual const std::vector<SubDeviceId>& get_sub_device_stall_group() const = 0;
    virtual void set_sub_device_stall_group(ttsl::Span<const SubDeviceId> sub_device_ids) = 0;
    virtual void reset_sub_device_stall_group() = 0;
    virtual uint32_t num_sub_devices() const = 0;
    virtual uint32_t num_virtual_eth_cores(SubDeviceId sub_device_id) = 0;

    // NOC Transaction Info
    virtual bool has_noc_mcast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const = 0;
    virtual uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool unicast_data = true) const = 0;

    // Misc
    virtual std::shared_ptr<distributed::MeshDevice> get_mesh_device() = 0;
};

}  // namespace tt::tt_metal
