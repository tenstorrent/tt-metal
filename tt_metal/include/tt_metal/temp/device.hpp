// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <utility>

// NEED TO REMOVE MOST OF THESE
// SOME WILL HAVE TO BECOME A PART OF PUBLIC API
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
#include "tt_metal/impl/device/program_cache.hpp"

namespace tt::tt_metal {

enum class BufferType;

inline namespace v0 {
// I'd prefer to avoid these fwd declares
// And instead include proper definitions
// But keeping this as is from the impl/device.hpp
class Buffer;
class Program;
class CommandQueue;
class SubDevice;

class IDevice {
public:
    virtual ~IDevice() = default;

    virtual tt::ARCH arch() const = 0;

    virtual CommandQueue& command_queue(size_t cq_id = 0) = 0;
    virtual SystemMemoryManager& sysmem_manager() = 0;

    virtual void run(std::function<void()>&& work, bool blocking = false) = 0;

    virtual void synchronize() = 0;
    virtual void enable_async(bool enable) = 0;

    virtual chip_id_t id() const = 0;
    virtual uint32_t build_key() const = 0;
    virtual uint8_t num_hw_cqs() const = 0;

    virtual bool is_initialized() const = 0;
    virtual int num_dram_channels() const = 0;

    virtual uint32_t l1_size_per_core() const = 0;
    virtual uint32_t dram_size_per_channel() const = 0;

    virtual CoreCoord grid_size() const = 0;
    virtual CoreCoord logical_grid_size() const = 0;
    virtual CoreCoord compute_with_storage_grid_size() const = 0;
    virtual CoreCoord dram_grid_size() const = 0;

    virtual CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const = 0;
    virtual CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;
    virtual CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const = 0;
    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;

    virtual std::vector<CoreCoord> ethernet_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const = 0;
    virtual std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment() = 0;

    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord, const CoreType& core_type) const = 0;
    virtual CoreCoord worker_core_from_logical_core(const CoreCoord& logical_core) const = 0;

    virtual CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical_core) const = 0;
    virtual CoreCoord logical_core_from_ethernet_core(const CoreCoord& ethernet_core) const = 0;

    virtual std::unordered_set<chip_id_t> get_ethernet_connected_device_ids() const = 0;
    virtual std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores = false) const = 0;

    virtual std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const = 0;
    virtual const std::vector<std::vector<chip_id_t>>& get_tunnels_from_mmio() const = 0;

    virtual std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const = 0;

    virtual bool is_mmio_capable() const = 0;

    virtual bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores = false) const = 0;
    virtual std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const = 0;
    virtual bool is_inactive_ethernet_core(CoreCoord logical_core) const = 0;

    virtual CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;
    virtual uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const = 0;

    virtual uint32_t num_sub_devices() const = 0;

    virtual uint32_t num_banks(const BufferType& buffer_type) const = 0;
    virtual uint32_t num_banks(const BufferType& buffer_type, SubDeviceId sub_device_id) const = 0;

    virtual uint32_t bank_size(const BufferType& buffer_type) const = 0;
    virtual uint32_t bank_size(const BufferType& buffer_type, SubDeviceId sub_device_id) const = 0;

    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id) const = 0;
    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const = 0;

    virtual CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const = 0;
    virtual uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const = 0;

    virtual allocator::Statistics get_memory_allocation_statistics(const BufferType& buffer_type) const = 0;
    virtual allocator::Statistics get_memory_allocation_statistics(
        const BufferType& buffer_type, SubDeviceId sub_device_id) const = 0;

    virtual const std::unordered_set<Buffer*>& get_allocated_buffers() const = 0;
    virtual const std::unordered_set<Buffer*>& get_allocated_buffers(SubDeviceId sub_device_id) const = 0;

    virtual DeviceAddr get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const = 0;
    virtual DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const = 0;
    virtual DeviceAddr get_base_allocator_addr(const HalMemType& mem_type, SubDeviceId sub_device_id) const = 0;

    virtual uint32_t get_allocator_alignment() const = 0;
    virtual uint32_t get_allocator_alignment(SubDeviceId sub_device_id) const = 0;

    virtual uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const = 0;
    virtual uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const = 0;

    virtual MemoryAllocator get_allocator_scheme() = 0;

    virtual const std::unique_ptr<Allocator>& get_initialized_allocator() const = 0;
    virtual const std::unique_ptr<Allocator>& get_initialized_allocator(SubDeviceId sub_device_id) const = 0;

    virtual void deallocate_buffers() = 0;
    virtual void deallocate_buffers(SubDeviceId sub_device_id) = 0;

    virtual float sfpu_eps() const = 0;
    virtual float sfpu_nan() const = 0;
    virtual float sfpu_inf() const = 0;

    virtual void generate_device_bank_to_noc_tables() = 0;

    virtual program_cache::detail::ProgramCache& get_program_cache() = 0;
    virtual void enable_program_cache() = 0;
    virtual void disable_and_clear_program_cache() = 0;
    virtual std::size_t num_program_cache_entries() = 0;

    virtual const std::vector<uint32_t>& bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const = 0;
    virtual const std::vector<uint32_t>& bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core, SubDeviceId sub_device_id) const = 0;

    virtual int32_t bank_offset(BufferType buffer_type, uint32_t bank_id) const = 0;
    virtual int32_t bank_offset(BufferType buffer_type, uint32_t bank_id, SubDeviceId sub_device_id) const = 0;

    // Metal trace device capture mode
    virtual void begin_trace(const uint8_t cq_id, const uint32_t tid) = 0;
    virtual void end_trace(const uint8_t cq_id, const uint32_t tid) = 0;
    virtual void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) = 0;
    virtual void release_trace(const uint32_t tid) = 0;
    virtual std::shared_ptr<TraceBuffer> get_trace(uint32_t tid) = 0;

    virtual WorkExecutorMode get_worker_mode() = 0;

    virtual bool close() = 0;
};

}  // namespace v0
}  // namespace tt::tt_metal
