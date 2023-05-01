#pragma once

#include <cstdint>
#include <vector>

#include "common/assert.hpp"
#include "common/tt_xy_pair.h"

namespace tt {

namespace tt_metal {

enum class MemoryAllocator {
    BASIC = 0,
    L1_BANKING = 1,
};

class Allocator {
   public:
    Allocator() {}

    ~Allocator() {}

    virtual uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual uint32_t get_address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank) const = 0;

    virtual void deallocate_dram_buffer(int dram_channel, uint32_t address) = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual uint32_t get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const = 0;

    virtual void deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address) = 0;

    virtual void clear_dram() = 0;

    virtual void clear_l1() = 0;

    virtual void clear() = 0;
};

namespace allocator {

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges);

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges
);

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
