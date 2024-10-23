// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include <unordered_set>

#include "allocator_types.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"
#include "llrt/hal.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
struct Allocator;

namespace allocator {

class BankManager {
   public:
    BankManager() {}

    BankManager(const BufferType &buffer_type, const std::vector<int64_t> &bank_descriptors, DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr alloc_offset=0);
    BankManager(const BufferType &buffer_type, const std::unordered_map<uint32_t, int64_t> &bank_id_to_descriptor, DeviceAddr size_bytes, DeviceAddr interleaved_address_limit, uint32_t alignment_bytes, DeviceAddr alloc_offset=0);
    BankManager&& operator=(BankManager&& that);
    ~BankManager();
    uint32_t num_banks() const;

    DeviceAddr bank_size() const;

    int64_t bank_offset(uint32_t bank_id) const;

    DeviceAddr allocate_buffer(DeviceAddr size, DeviceAddr page_size, bool bottom_up, CoreCoord compute_grid_size, std::optional<uint32_t> num_shards);

    void deallocate_buffer(DeviceAddr address);
    void deallocate_all();

    void clear();

    std::optional<DeviceAddr> lowest_occupied_address(uint32_t bank_id) const;

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream &out) const;

   private:
    void deallocate_buffer_(DeviceAddr address);

    // Types of buffers allocated in the banks
    BufferType buffer_type_;
    std::unordered_set<DeviceAddr> allocated_buffers_;
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all banks using only bank_id
    // Set to 0 for cores/nodes with only 1 bank
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_;
    std::unique_ptr<Algorithm> allocator_;
    DeviceAddr interleaved_address_limit_;
    uint32_t alignment_bytes_;
    void validate_bank_id(uint32_t bank_id) const;

    void init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset);
};

DeviceAddr get_unreserved_base_address(const Allocator &allocator, const HalMemType &mem_type);

// Functions used to initiate allocator and allocate buffers
void init_one_bank_per_channel(Allocator &allocator, const AllocatorConfig &alloc_config);

void init_one_bank_per_l1(Allocator &allocator, const AllocatorConfig &alloc_config);

uint32_t num_banks(const Allocator &allocator, const BufferType &buffer_type);

DeviceAddr bank_size(const Allocator &allocator, const BufferType &buffer_type);

uint32_t dram_channel_from_bank_id(const Allocator &allocator, uint32_t bank_id);

CoreCoord logical_core_from_bank_id(const Allocator &allocator, uint32_t bank_id);

int32_t bank_offset(const Allocator &allocator, BufferType buffer_type, uint32_t bank_id);

const std::vector<uint32_t> &bank_ids_from_dram_channel(const Allocator &allocator, uint32_t dram_channel);

const std::vector<uint32_t> &bank_ids_from_logical_core(
    const Allocator &allocator, BufferType buffer_type, const CoreCoord &logical_core);

Statistics get_statistics(const Allocator &allocator, const BufferType &buffer_type);

void dump_memory_blocks(const Allocator &allocator, const BufferType &buffer_type, std::ofstream &out);

std::optional<DeviceAddr> lowest_occupied_l1_address(const Allocator &allocator, uint32_t bank_id);

DeviceAddr base_alloc(const AllocatorConfig & config, BankManager &bank_manager, DeviceAddr size, DeviceAddr page_size, bool bottom_up, std::optional<uint32_t> num_shards);

DeviceAddr allocate_buffer(Allocator &allocator, DeviceAddr size, DeviceAddr page_size, const BufferType &buffer_type, bool bottom_up, std::optional<uint32_t> num_shards = std::nullopt);

void mark_allocations_unsafe(Allocator &allocator);

void mark_allocations_safe(Allocator &allocator);

void deallocate_buffer(Allocator &allocator, DeviceAddr address, const BufferType &buffer_type);
void deallocate_buffers(Allocator &allocator);

void clear(Allocator &allocatator);

}  // namespace allocator

struct Allocator {
    Allocator(const AllocatorConfig &alloc_config, const allocator::AllocDescriptor &alloc_descriptor);
    // Set to true if allocating a buffer is unsafe. This happens when a live trace on device can corrupt
    // memory allocated by the user (memory used by trace is not tracked in the allocator once the trace is captured).
    bool allocations_unsafe = false;
    allocator::BankManager dram_manager;
    allocator::BankManager l1_manager;
    allocator::BankManager l1_small_manager;
    allocator::BankManager trace_buffer_manager;
    // TODO: Track lowest l1 addresses!

    std::unordered_map<uint32_t, uint32_t> bank_id_to_dram_channel;
    std::unordered_map<uint32_t, std::vector<uint32_t>> dram_channel_to_bank_ids;
    std::unordered_map<uint32_t, CoreCoord> bank_id_to_logical_core;
    std::unordered_map<BufferType, std::unordered_map<CoreCoord, std::vector<uint32_t>>> logical_core_to_bank_ids;

    AllocatorConfig config;
    // Callbacks to invoke during initialization and allocation
    allocator::AllocDescriptor descriptor;

    void reset();
    ~Allocator();
};

}  // namespace tt_metal

}  // namespace tt
