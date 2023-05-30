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

inline bool accept_all_address_ranges(const std::pair<u32, u32> &range) { return true; }

inline void pass_through_potential_addresses(u32 bank_id, std::vector<std::pair<u32, u32>> &potential_addr_ranges) {}

inline u32 pass_through_adjust_address(u32 address, u32 bank_id) { return address; }

class BankManager {
   public:
    BankManager() : initialized_(false) {}

    BankManager(const std::vector<BankDescriptor> &bank_descriptors);
    BankManager(const std::unordered_map<u32, BankDescriptor> &bank_id_to_descriptor);

    u32 num_banks() const;

    u32 size(u32 bank_id) const;

    u32 offset(u32 bank_id) const;
    i32 l1_bank_offset(u32 bank_id) const;

    BankIdToRelativeAddress allocate_buffer(
        u32 starting_bank_id, u32 size, u32 page_size, bool bottom_up,
        std::function<void(u32, std::vector<std::pair<u32, u32>> &)> adjust_potential_addresses = pass_through_potential_addresses,
        std::function<bool(const std::pair<u32, u32> &)> filter = accept_all_address_ranges,
        std::function<u32(u32, u32)> adjust_relative_address = pass_through_adjust_address);

    BankIdToRelativeAddress allocate_buffer_at_address(
        u32 starting_bank_id, u32 size, u32 page_size, u32 absolute_address,
        std::function<u32(u32, u32)> adjust_absolute_address = pass_through_adjust_address);

    void deallocate_buffer(u32 bank_id, u32 absolute_address);

    // TODO (abhullar): Remove after CB redesign
    std::vector<std::pair<u32, u32>> available_addresses(u32 bank_id, u32 size_bytes, bool return_absolute_addresses=false) const;

    void clear();

   private:
    constexpr static u32 min_allocation_size_bytes_ = 32;
    // DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
    constexpr static u32 alignment_ = 32;

    bool initialized_;
    std::unordered_map<u32, u32> bank_id_to_offset_;
    std::unordered_map<u32, u32> bank_id_to_l1_bank_offset_;
    std::unordered_map<u32, std::unique_ptr<Algorithm>> bank_id_to_allocator_;

    void validate_bank_id(u32 bank_id) const;

    BankIdToRelativeAddress allocate_contiguous_buffer(u32 bank_id, u32 size_bytes, bool bottom_up);

    BankIdToRelativeAddress allocate_contiguous_buffer_at_address(u32 bank_id, u32 size_bytes, u32 address);
};

u32 find_max_address(const std::vector<std::pair<u32, u32>> &candidate_addr_ranges);

u32 find_address_of_smallest_chunk(const std::vector<std::pair<u32, u32>> &candidate_addr_ranges);

void populate_candidate_address_ranges(
    std::vector<std::pair<u32, u32>> &candidate_addr_ranges,
    const std::vector<std::pair<u32, u32>> &potential_addr_ranges,
    std::function<bool(const std::pair<u32, u32> &)> filter = accept_all_address_ranges
);

// Functions used to initiate allocator and allocate buffers
void init_one_bank_per_channel(Allocator &allocator, const AllocatorConfig &alloc_config);

void init_one_bank_per_l1(Allocator &allocator, const AllocatorConfig &alloc_config);

u32 num_banks(const Allocator &allocator, const BufferType &buffer_type);

u32 dram_channel_from_bank_id(const Allocator &allocator, u32 bank_id);

CoreCoord logical_core_from_bank_id(const Allocator &allocator, u32 bank_id);

i32 l1_bank_offset_from_bank_id(const Allocator &allocator, u32 bank_id);

std::vector<u32> bank_ids_from_dram_channel(const Allocator &allocator, u32 dram_channel);

std::vector<u32> bank_ids_from_logical_core(const Allocator &allocator, const CoreCoord &logical_core);

BankIdToRelativeAddress alloc_one_bank_per_storage_unit(const AllocatorConfig & config, BankManager &bank_manager, u32 starting_bank_id, u32 size, u32 page_size, bool bottom_up);

BankIdToRelativeAddress alloc_at_addr_one_bank_per_storage_unit(const AllocatorConfig & config, BankManager &bank_manager, u32 starting_bank_id, u32 size, u32 page_size, u32 absolute_address);

BankIdToRelativeAddress allocate_buffer(Allocator &allocator, u32 starting_bank_id, u32 size, u32 page_size, const BufferType &buffer_type, bool bottom_up);

BankIdToRelativeAddress allocate_buffer_at_address(Allocator &allocator, u32 starting_bank_id, u32 size, u32 page_size, u32 absolute_address, const BufferType &buffer_type);

void deallocate_buffer(Allocator &allocator, u32 bank_id, u32 address, const BufferType &buffer_type);

void clear(Allocator &allocatator);

// TODO (abhullar): Circular buffer specific APIs will be removed after CB redesign.
u32 allocate_circular_buffer(Allocator &allocator, const CoreCoord &logical_core, u32 size_bytes);

u32 allocate_circular_buffer(Allocator &allocator, const CoreCoord &logical_core, u32 start_address, u32 size_bytes);

u32 get_address_for_circular_buffers_across_core_range(Allocator &allocator, const CoreRange &logical_core_range, u32 size_in_bytes);

}  // namespace allocator

struct Allocator {
    Allocator(const AllocatorConfig &alloc_config, const allocator::AllocDescriptor &alloc_descriptor);

    allocator::BankManager dram_manager;
    allocator::BankManager l1_manager;

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
