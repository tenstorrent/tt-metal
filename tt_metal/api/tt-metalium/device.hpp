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
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include <tt_stl/span.hpp>

namespace tt {

namespace tt_metal {

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
class Buffer;
class Program;
class SubDevice;

class CommandQueue;
class SystemMemoryManager;
struct TraceBuffer;
struct TraceDescriptor;

namespace distributed {
class MeshDevice;
}

class IDeviceImpl;

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

    virtual int num_dram_channels() const = 0;
    virtual uint32_t l1_size_per_core() const = 0;
    virtual CoreCoord grid_size() const = 0;

    // Returns the optimal DRAM bank coordinates to logical worker assignment based on which noc will be issuing DRAM
    // requests
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(NOC noc) = 0;

    // Convert a logical coordinate to virtual coordinate
    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord, const CoreType& core_type) const = 0;

    // Convert a logical coordinate to a virtual coordinate for a worker coordinate
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const = 0;

    // Returns virtual coordinates of active ethernet cores. Some ethernet cores may be reserved for dispatch use.
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;

    virtual std::vector<CoreCoord> get_ethernet_sockets(ChipId connected_chip_id) const = 0;

    virtual CoreCoord compute_with_storage_grid_size() const = 0;

    // Returns a logical CoreRangeSet of the worker cores in the specified sub device that was previously loaded.
    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;
    // Returns the number of worker cores in the specified sub device that was previously loaded.
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    virtual const std::unique_ptr<Allocator>& allocator() const = 0;

    virtual std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const = 0;

    [[deprecated(
        "Storage-only cores do not exist. Cleanup code that calls this API.")]] virtual const std::set<CoreCoord>&
    storage_only_cores() const = 0;

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    virtual void enable_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    void set_program_cache_misses_allowed(bool allowed);

    virtual const std::vector<SubDeviceId>& get_sub_device_ids() const = 0;

    virtual bool is_mmio_capable() const = 0;

    virtual IDeviceImpl* impl() = 0;
    virtual const IDeviceImpl* impl() const = 0;
};

}  // namespace tt_metal

}  // namespace tt
