#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "common/assert.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/common/tt_soc_descriptor.h"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
struct Allocator;

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

struct AddressDescriptor {
    uint32_t offset_bytes;
    uint32_t relative_address;

    uint32_t absolute_address()     const {
        return offset_bytes + relative_address;
    }
};

using BankIdToRelativeAddress = std::unordered_map<uint32_t, AddressDescriptor>;

struct BankDescriptor {
    uint32_t offset_bytes;
    uint32_t size_bytes;
};

namespace allocator {

inline bool accept_all_address_ranges(const std::pair<uint32_t, uint32_t> &range) { return true; }

inline void pass_through_potential_addresses(uint32_t bank_id, std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges) {}

inline uint32_t pass_through_adjust_address(uint32_t address, uint32_t bank_id) { return address; }

class BankManager {
   public:
    BankManager() : initialized_(false) {}

    BankManager(uint32_t num_banks, uint32_t offset_bytes, uint32_t bank_size_bytes);

    BankManager(const std::unordered_map<uint32_t, BankDescriptor> &bank_id_to_descriptor);

    uint32_t num_banks() const;

    uint32_t size(uint32_t bank_id) const;

    uint32_t offset(uint32_t bank_id) const;

    BankIdToRelativeAddress allocate_buffer(
        uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up,
        std::function<void(uint32_t, std::vector<std::pair<uint32_t, uint32_t>> &)> adjust_potential_addresses = pass_through_potential_addresses,
        std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter = accept_all_address_ranges,
        std::function<uint32_t(uint32_t, uint32_t)> adjust_relative_address = pass_through_adjust_address);

    BankIdToRelativeAddress allocate_buffer_at_address(
        uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address,
        std::function<uint32_t(uint32_t, uint32_t)> adjust_absolute_address = pass_through_adjust_address);

    void deallocate_buffer(uint32_t bank_id, uint32_t absolute_address);

    // TODO (abhullar): Remove after CB redesign
    std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t bank_id, uint32_t size_bytes, bool return_absolute_addresses=false) const;

    void clear();

   private:
    constexpr static uint32_t min_allocation_size_bytes_ = 32;
    // DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
    constexpr static uint32_t alignment_ = 32;

    bool initialized_;
    std::unordered_map<uint32_t, uint32_t> bank_id_to_offset_;
    std::unordered_map<uint32_t, std::unique_ptr<Algorithm>> bank_id_to_allocator_;

    void validate_bank_id(uint32_t bank_id) const;

    BankIdToRelativeAddress allocate_contiguous_buffer(uint32_t bank_id, uint32_t size_bytes, bool bottom_up);

    BankIdToRelativeAddress allocate_contiguous_buffer_at_address(uint32_t bank_id, uint32_t size_bytes, uint32_t address);
};

uint32_t find_max_address(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges);

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges);

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges,
    std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter = accept_all_address_ranges
);

struct InitAndAllocFuncs {
    std::function<void(Allocator &, const tt_SocDescriptor &)> init;
    std::function<BankIdToRelativeAddress(BankManager &, uint32_t, uint32_t, uint32_t, bool)> alloc;
    std::function<BankIdToRelativeAddress(BankManager &, uint32_t, uint32_t, uint32_t, uint32_t)> alloc_at_addr;
};

// Holds callback functions required by allocators that specify how to initialize the bank managers and what the allocation scheme
// is for a given storage substrate
struct AllocDescriptor {
    InitAndAllocFuncs dram;
    InitAndAllocFuncs l1;
};

void init_one_bank_per_channel(Allocator &allocator, const tt_SocDescriptor &soc_desc);

void init_one_bank_per_l1(Allocator &allocator, const tt_SocDescriptor &soc_desc);

uint32_t num_banks(const Allocator &allocator, const BufferType &buffer_type);

uint32_t dram_channel_from_bank_id(const Allocator &allocator, uint32_t bank_id);

tt_xy_pair logical_core_from_bank_id(const Allocator &allocator, uint32_t bank_id);

std::vector<uint32_t> bank_ids_from_dram_channel(const Allocator &allocator, uint32_t dram_channel);

std::vector<uint32_t> bank_ids_from_logical_core(const Allocator &allocator, const tt_xy_pair &logical_core);

BankIdToRelativeAddress alloc_one_bank_per_storage_unit(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, bool bottom_up);

BankIdToRelativeAddress alloc_at_addr_one_bank_per_storage_unit(BankManager &bank_manager, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address);

BankIdToRelativeAddress allocate_buffer(Allocator &allocator, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, const BufferType &buffer_type, bool bottom_up);

BankIdToRelativeAddress allocate_buffer_at_address(Allocator &allocator, uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t absolute_address, const BufferType &buffer_type);

void deallocate_buffer(Allocator &allocator, uint32_t bank_id, uint32_t address, const BufferType &buffer_type);

void clear(Allocator &allocatator);

// TODO (abhullar): Circular buffer specific APIs will be removed after CB redesign.
uint32_t allocate_circular_buffer(Allocator &allocator, const tt_xy_pair &logical_core, uint32_t size_bytes);

uint32_t allocate_circular_buffer(Allocator &allocator, const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes);

uint32_t get_address_for_circular_buffers_across_core_range(Allocator &allocator, const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes);

}  // namespace allocator

struct Allocator {
    Allocator(const tt_SocDescriptor &soc_desc, const allocator::AllocDescriptor &alloc_descriptor);

    allocator::BankManager dram_manager;
    allocator::BankManager l1_manager;

    std::unordered_map<uint32_t, uint32_t> bank_id_to_dram_channel;
    std::unordered_map<uint32_t, std::vector<uint32_t>> dram_channel_to_bank_ids;
    std::unordered_map<uint32_t, tt_xy_pair> bank_id_to_logical_core;
    std::unordered_map<tt_xy_pair, std::vector<uint32_t>> logical_core_to_bank_ids;

    // Callbacks to invoke during initialization and allocation
    allocator::AllocDescriptor descriptor;
};

}  // namespace tt_metal

}  // namespace tt
