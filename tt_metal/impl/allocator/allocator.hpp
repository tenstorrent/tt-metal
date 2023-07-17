#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "allocator_types.hpp"
#include "common/assert.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
struct Allocator;

namespace allocator {

inline u32 pass_through_address(u32 address) { return address; }

class BankManager {
   public:
    BankManager() {}

    BankManager(const std::vector<i64> &bank_descriptors, u64 size_bytes, u64 alloc_offset=0);
    BankManager(const std::unordered_map<u32, i64> &bank_id_to_descriptor, u64 size_bytes, u64 alloc_offset=0);

    u32 num_banks() const;

    i64 bank_offset(u32 bank_id) const;

    u64 allocate_buffer(u32 size, u32 page_size, bool bottom_up);

    u64 allocate_buffer_at_address(u32 size, u32 page_size, u32 relative_address, std::function<u32(u32)> adjust_address = pass_through_address);

    void deallocate_buffer(u64 address);

    void clear();

    std::optional<u64> lowest_occupied_address(u32 bank_id) const;

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream &out) const;

   private:
    constexpr static u32 min_allocation_size_bytes_ = 32;

    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all banks using only bank_id
    // Set to 0 for cores/nodes with only 1 bank
    std::unordered_map<u32, i64> bank_id_to_bank_offset_;
    std::unique_ptr<Algorithm> allocator_;

    void validate_bank_id(u32 bank_id) const;

    void init_allocator(u64 size_bytes, u64 offset);
};

// Functions used to initiate allocator and allocate buffers
void init_one_bank_per_channel(Allocator &allocator, const AllocatorConfig &alloc_config);

void init_one_bank_per_l1(Allocator &allocator, const AllocatorConfig &alloc_config);

u32 num_banks(const Allocator &allocator, const BufferType &buffer_type);

u32 dram_channel_from_bank_id(const Allocator &allocator, u32 bank_id);

CoreCoord logical_core_from_bank_id(const Allocator &allocator, u32 bank_id);

i32 l1_bank_offset_from_bank_id(const Allocator &allocator, u32 bank_id);

i32 dram_bank_offset_from_bank_id(const Allocator &allocator, u32 bank_id);

std::vector<u32> bank_ids_from_dram_channel(const Allocator &allocator, u32 dram_channel);

std::vector<u32> bank_ids_from_logical_core(const Allocator &allocator, const CoreCoord &logical_core);

Statistics get_statistics(const Allocator &allocator, const BufferType &buffer_type);

void dump_memory_blocks(const Allocator &allocator, const BufferType &buffer_type, std::ofstream &out);

std::optional<u64> lowest_occupied_l1_address(const Allocator &allocator, u32 bank_id);

u64 base_alloc(const AllocatorConfig & config, BankManager &bank_manager, u64 size, u64 page_size, bool bottom_up);

u64 base_alloc_at_addr(const AllocatorConfig &config, BankManager &bank_manager, u64 size, u64 page_size, u64 absolute_address);

u64 allocate_buffer(Allocator &allocator, u32 size, u32 page_size, const BufferType &buffer_type, bool bottom_up);

u64 allocate_buffer_at_address(Allocator &allocator, u32 size, u32 page_size, u32 relative_address, const BufferType &buffer_type);

void deallocate_buffer(Allocator &allocator, u64 address, const BufferType &buffer_type);

void clear(Allocator &allocatator);

}  // namespace allocator

struct Allocator {
    Allocator(const AllocatorConfig &alloc_config, const allocator::AllocDescriptor &alloc_descriptor);

    allocator::BankManager dram_manager;
    allocator::BankManager l1_manager;

    // TODO: Track lowest l1 addresses!

    std::unordered_map<u32, u32> bank_id_to_dram_channel;
    std::unordered_map<u32, std::vector<u32>> dram_channel_to_bank_ids;
    std::unordered_map<u32, CoreCoord> bank_id_to_logical_core;
    std::unordered_map<CoreCoord, std::vector<u32>> logical_core_to_bank_ids;

    AllocatorConfig config;
    // Callbacks to invoke during initialization and allocation
    allocator::AllocDescriptor descriptor;
};

}  // namespace tt_metal

}  // namespace tt
