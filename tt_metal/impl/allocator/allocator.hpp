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

struct DramBank {
    int channel;
    uint32_t offset_bytes;
};

constexpr inline bool operator==(const DramBank &a, const DramBank &b) { return a.channel == b.channel && a.offset_bytes == b.offset_bytes; }

constexpr inline bool operator!=(const DramBank &a, const DramBank &b) { return !(a == b); }

constexpr inline bool operator<(const DramBank &a, const DramBank &b) {
    return (a.channel < b.channel) or (a.channel == b.channel and a.offset_bytes < b.offset_bytes);
}

struct L1Bank {
    tt_xy_pair logical_core;
    uint32_t offset_bytes;
};

constexpr inline bool operator==(const L1Bank &a, const L1Bank &b) { return a.logical_core == b.logical_core && a.offset_bytes == b.offset_bytes; }

constexpr inline bool operator!=(const L1Bank &a, const L1Bank &b) { return !(a == b); }

constexpr inline bool operator<(const L1Bank &a, const L1Bank &b) {
    return (a.logical_core < b.logical_core) or (a.logical_core == b.logical_core and a.offset_bytes < b.offset_bytes);
}

using DramBankAddrPair = std::pair<DramBank, uint32_t>;
using L1BankAddrPair = std::pair<L1Bank, uint32_t>;

class Allocator {
   public:
    Allocator() {}

    ~Allocator() {}

    virtual uint32_t allocate_dram_buffer(int dram_channel, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual std::vector<DramBankAddrPair> allocate_interleaved_dram_buffer(int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) = 0;

    virtual void deallocate_dram_buffer(int dram_channel, uint32_t address) = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_circular_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) = 0;

    virtual uint32_t allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) = 0;

    virtual std::vector<L1BankAddrPair> allocate_interleaved_l1_buffer(int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry) = 0;

    virtual uint32_t get_address_for_circular_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const = 0;

    virtual void deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address) = 0;

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
