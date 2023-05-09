#pragma once

#include <cstdint>
#include <vector>

#include "common/assert.hpp"
#include "common/tt_xy_pair.h"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

struct Address {
    uint32_t offset_bytes;
    uint32_t relative_address;

    uint32_t absolute_address() const {
        return offset_bytes + relative_address;
    }
};

using BankIdToRelativeAddress = std::unordered_map<uint32_t, Address>;

class Allocator {
   public:
    Allocator() {}

    ~Allocator() {}

    virtual BankIdToRelativeAddress allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, const BufferType &buffer_type) = 0;

    virtual BankIdToRelativeAddress allocate_buffer(uint32_t starting_bank_id, uint32_t size, uint32_t page_size, uint32_t address, const BufferType &buffer_type) = 0;

    virtual void deallocate_buffer(uint32_t bank_id, uint32_t address, const BufferType &buffer_type) = 0;

    virtual uint32_t num_banks(const BufferType &buffer_type) const = 0;

    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id) const = 0;

    virtual tt_xy_pair logical_core_from_bank_id(uint32_t bank_id) const = 0;

    virtual std::vector<uint32_t> bank_ids_from_dram_channel(uint32_t dram_channel) const = 0;

    virtual std::vector<uint32_t> bank_ids_from_logical_core(const tt_xy_pair &logical_core) const = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual uint32_t get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const = 0;

    virtual void clear_dram() = 0;

    virtual void clear_l1() = 0;

    virtual void clear() = 0;
};

namespace allocator {

uint32_t find_max_address(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges);

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges);

inline bool accept_all_address_ranges(const std::pair<uint32_t, uint32_t> &range) { return true; }

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges,
    std::function<bool(const std::pair<uint32_t, uint32_t> &)> filter = accept_all_address_ranges
);

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
