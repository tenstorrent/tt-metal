// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>

// UMD: re-exports tt::ARCH
#include <umd/device/types/arch.hpp>
#include <tt-metalium/device_types.hpp>
// UMD: re-exports CoreType (used in IDevice::virtual_core_from_logical_core parameter).
#include <umd/device/types/core_coordinates.hpp>

#include <tt_stl/span.hpp>

#include <tt-metalium/device_internal.hpp>

namespace tt::tt_metal {

// Forward declaration
enum NOC : uint8_t;

namespace program_cache::detail {
struct ProgramCache;
}
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;
enum class BufferType;

class Allocator;
class AllocatorImpl;
class Buffer;
class Program;
class SubDevice;

class SystemMemoryManager;
struct TraceBuffer;
struct TraceDescriptor;

namespace distributed {
class MeshDevice;
}

class IDevice {
public:
    IDevice() = default;
    virtual ~IDevice() = default;

    IDevice(const IDevice& other) = delete;
    IDevice& operator=(const IDevice& other) = delete;

    IDevice(IDevice&& other) = default;
    IDevice& operator=(IDevice&& /*other*/) = default;

    virtual tt::ARCH arch() const = 0;

    virtual ChipId id() const = 0;

    // Access internal-only methods via device_internal().
    // These methods are used internally by tt_metal and tests.
    virtual IDeviceInternal& device_internal() = 0;
    virtual const IDeviceInternal& device_internal() const = 0;

    virtual int num_dram_channels() const = 0;
    virtual uint32_t l1_size_per_core() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;
    // Returns the AI clock frequency in MHz for this device.
    // This value is queried from the actual hardware via the cluster API
    // and reflects the device's current operating frequency.
    virtual int get_clock_rate_mhz() const = 0;
    virtual CoreCoord grid_size() const = 0;
    virtual CoreCoord logical_grid_size() const = 0;
    virtual CoreCoord dram_grid_size() const = 0;

    // Returns the optimal DRAM bank coordinates to logical worker assignment based on which noc will be issuing DRAM
    // requests
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) = 0;

    // Convert a logical coordinate to virtual coordinate
    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord, const CoreType& core_type) const = 0;

    // Convert a logical coordinate to a virtual coordinate for a worker coordinate
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const = 0;

    virtual CoreCoord compute_with_storage_grid_size() const = 0;

    // Returns a logical CoreRangeSet of the worker cores in the specified sub device that was previously loaded.
    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    virtual const std::unique_ptr<Allocator>& allocator() const = 0;
    virtual const std::unique_ptr<Allocator>& allocator(SubDeviceId sub_device_id) const = 0;

    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const = 0;
    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address(
        ttsl::Span<const SubDeviceId> sub_device_ids) const = 0;

    [[deprecated(
        "Storage-only cores do not exist. Cleanup code that calls this API.")]] virtual const std::set<CoreCoord>&
    storage_only_cores() const = 0;

    // Puts device into reset
    virtual bool close() = 0;

    // Program cache interface. Synchronize with worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    virtual void enable_program_cache() = 0;
    virtual void clear_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    void set_program_cache_misses_allowed(bool allowed);
    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    virtual SubDeviceManagerId get_active_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId get_default_sub_device_manager_id() const = 0;
    virtual SubDeviceManagerId create_sub_device_manager(
        ttsl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual SubDeviceManagerId create_sub_device_manager(
        std::initializer_list<SubDevice> sub_devices, DeviceAddr local_l1_size) = 0;
    virtual void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) = 0;
    virtual void clear_loaded_sub_device_manager() = 0;
    virtual const std::vector<SubDeviceId>& get_sub_device_ids() const = 0;
    virtual void set_sub_device_stall_group(ttsl::Span<const SubDeviceId> sub_device_ids) = 0;
    virtual void reset_sub_device_stall_group() = 0;

    virtual bool is_mmio_capable() const = 0;

    virtual std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const = 0;
};

}  // namespace tt::tt_metal
