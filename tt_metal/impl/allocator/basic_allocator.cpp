#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

#include <cmath>

namespace tt {

namespace tt_metal {

BasicAllocator::BasicAllocator(const tt_SocDescriptor &soc_desc) : Allocator() {
    constexpr static uint32_t min_allocation_size_bytes = 32;
    // DRAM -> L1 and L1 -> DRAM transfers need to have 32B alignment, which means:
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32, or
    // DRAM_buffer_addr % 32 == L1_buffer_addr % 32 == 0
    constexpr static uint32_t alignment = 32;

    // Initialize DRAM manager
    for (int dram_channel = 0; dram_channel < soc_desc.get_num_dram_channels(); dram_channel++) {
        auto allocator = std::make_unique<allocator::FreeList>(
            soc_desc.dram_bank_size,
            min_allocation_size_bytes,
            alignment,
            allocator::FreeList::SearchPolicy::FIRST
        );
        this->dram_manager_.insert({dram_channel, std::move(allocator)});
    }

    for (auto worker_core : soc_desc.workers) {
        auto logical_core_x = soc_desc.routing_x_to_worker_x.at(worker_core.x);
        auto logical_core_y = soc_desc.routing_y_to_worker_y.at(worker_core.y);
        tt_xy_pair logical_core = tt_xy_pair(logical_core_x, logical_core_y);
        auto allocator = std::make_unique<allocator::FreeList>(
            soc_desc.worker_l1_size,
            min_allocation_size_bytes,
            alignment,
            allocator::FreeList::SearchPolicy::FIRST
        );
        // Space up to UNRESERVED_BASE is reserved for risc binaries, kernel args, debug and perf monitoring tools
        allocator->allocate(0, UNRESERVED_BASE);
        this->l1_manager_.insert({logical_core, std::move(allocator)});
    }
}

// BasicAllocator::BasicAllocator(const BasicAllocator &other);
// BasicAllocator& operator=(const BasicAllocator &other);

// BasicAllocator(BasicAllocator &&other);
// BasicAllocator& operator=(BasicAllocator &&other);

allocator::Algorithm &BasicAllocator::allocator_for_dram_channel(int dram_channel) const {
    if (this->dram_manager_.find(dram_channel) == this->dram_manager_.end()) {
        TT_THROW("Allocator for DRAM channel " + std::to_string(dram_channel) + " does not exist!");
    }
    return *this->dram_manager_.at(dram_channel);
}

allocator::Algorithm &BasicAllocator::allocator_for_logical_core(const tt_xy_pair &logical_core) const {
    if (this->l1_manager_.find(logical_core) == this->l1_manager_.end()) {
        TT_THROW("Allocator for logical core " + logical_core.str() + " does not exist!");
    }
    return *this->l1_manager_.at(logical_core);
}

uint32_t BasicAllocator::allocate_dram_buffer(int dram_channel, uint32_t size_bytes) {
    return this->allocator_for_dram_channel(dram_channel).allocate(size_bytes);
}

uint32_t BasicAllocator::allocate_dram_buffer(int dram_channel, uint32_t start_address, uint32_t size_bytes) {
    return this->allocator_for_dram_channel(dram_channel).allocate(start_address, size_bytes);
}

uint32_t find_address_of_smallest_chunk(const std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges) {
    uint32_t smallest_chunk = candidate_addr_ranges[0].second - candidate_addr_ranges[0].first;
    uint32_t address = candidate_addr_ranges[0].first;
    for (auto candidate_addr_range : candidate_addr_ranges) {
        uint32_t range_size = candidate_addr_range.second - candidate_addr_range.first;
        if (range_size < smallest_chunk) {
            smallest_chunk = range_size;
            address = candidate_addr_range.first;
        }
    }
    return address;
}

void populate_candidate_address_ranges(
    std::vector<std::pair<uint32_t, uint32_t>> &candidate_addr_ranges,
    const std::vector<std::pair<uint32_t, uint32_t>> &potential_addr_ranges) {
    if (candidate_addr_ranges.empty()) {
        candidate_addr_ranges = potential_addr_ranges;
        return;
    }
    int i = 0;
    int j = 0;
    std::vector<std::pair<uint32_t, uint32_t>> intersecting_addr_ranges;
    while (i < candidate_addr_ranges.size() and j < potential_addr_ranges.size()) {
        uint32_t lower_addr = std::max(candidate_addr_ranges[i].first, potential_addr_ranges[j].first);
        uint32_t upper_addr = std::min(candidate_addr_ranges[i].second, potential_addr_ranges[j].second);
        if (lower_addr <= upper_addr) {
            intersecting_addr_ranges.push_back({lower_addr, upper_addr});
        }
        if (candidate_addr_ranges[i].second < potential_addr_ranges[j].second) {
            i++;
        } else {
            j++;
        }
    }
    candidate_addr_ranges = intersecting_addr_ranges;
}

uint32_t BasicAllocator::get_address_for_interleaved_dram_buffer(const std::map<int, uint32_t> &size_in_bytes_per_bank) const {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    uint32_t total_size_bytes = 0;
    for (auto &[dram_channel, required_size_bytes] : size_in_bytes_per_bank) {
        auto potential_addr_ranges = this->allocator_for_dram_channel(dram_channel).available_addresses(required_size_bytes);
        populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        total_size_bytes += required_size_bytes;
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space to hold interleave " + std::to_string(total_size_bytes) + " bytes across DRAM channels");
    }

    return find_address_of_smallest_chunk(candidate_addr_ranges);
}

void BasicAllocator::deallocate_dram_buffer(int dram_channel, uint32_t address) {
    this->allocator_for_dram_channel(dram_channel).deallocate(address);
}

uint32_t BasicAllocator::allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t size_bytes) {
    return this->allocator_for_logical_core(logical_core).allocate(size_bytes);
}

uint32_t BasicAllocator::allocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t start_address, uint32_t size_bytes) {
    return this->allocator_for_logical_core(logical_core).allocate(start_address, size_bytes);
}

void BasicAllocator::deallocate_l1_buffer(const tt_xy_pair &logical_core, uint32_t address) {
    this->allocator_for_logical_core(logical_core).deallocate(address);
}

uint32_t BasicAllocator::get_address_for_l1_buffers_across_core_range(const std::pair<tt_xy_pair, tt_xy_pair> &logical_core_range, uint32_t size_in_bytes) const {
    std::vector<std::pair<uint32_t, uint32_t>> candidate_addr_ranges;
    auto start_core = logical_core_range.first;
    auto end_core = logical_core_range.second;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            auto potential_addr_ranges = this->allocator_for_logical_core(logical_core).available_addresses(size_in_bytes);
            populate_candidate_address_ranges(candidate_addr_ranges, potential_addr_ranges);
        }
    }

    if (candidate_addr_ranges.empty()) {
        TT_THROW("Not enough space for " + std::to_string(size_in_bytes) +
            " byte CircularBuffers in cores ranging from " + start_core.str() + " to " + end_core.str());
    }

    return find_address_of_smallest_chunk(candidate_addr_ranges);
}

void BasicAllocator::clear_dram() {
    for (auto &[dram_channel, dram_allocator] : this->dram_manager_) {
        dram_allocator->clear();
    }
}

void BasicAllocator::clear_l1() {
    for (auto &[logical_core, l1_allocator] : this->l1_manager_) {
        l1_allocator->clear();
    }
}

void BasicAllocator::clear() {
    this->clear_dram();
    this->clear_l1();
}

}  // namespace tt_metal

}  // namespace tt
