// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>

#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {
namespace tt_metal {
namespace allocator {
// Essentially the same free list algorithm as FreeList with BestFit policy, but with (IMO absurdly) optimized code.
// Including
// - SoA instead of linked list for the free list
// - Size segregated to avoid unnecessary searches of smaller blocks
// - Hash table to store allocated blocks for faster block lookup during deallocation
// - Keeps metadata locality to avoid cache misses
// - Metadata reuse to avoid allocations
class FreeListOpt : public Algorithm {
public:
    enum class SearchPolicy {
        FIRST,
        BEST,
    };
    FreeListOpt(
        DeviceAddr max_size_bytes,
        DeviceAddr offset_bytes,
        DeviceAddr min_allocation_size,
        DeviceAddr alignment,
        SearchPolicy policy = SearchPolicy::BEST);
    void init() override;

    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_addresses(DeviceAddr size_bytes) const override;

    std::optional<DeviceAddr> allocate(
        DeviceAddr size_bytes, bool bottom_up = true, DeviceAddr address_limit = 0) override;

    std::optional<DeviceAddr> allocate_at_address(DeviceAddr absolute_start_address, DeviceAddr size_bytes) override;

    void deallocate(DeviceAddr absolute_address) override;

    void clear() override;

    Statistics get_statistics() const override;

    void dump_blocks(std::ostream& out) const override;

    std::vector<std::unordered_map<std::string, std::string>> get_block_table() const override;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true) override;

    void reset_size() override;

private:
    // SoA free list components
    std::vector<DeviceAddr> block_address_;
    std::vector<DeviceAddr> block_size_;
    std::vector<ssize_t> block_prev_block_;
    std::vector<ssize_t> block_next_block_;
    std::vector<uint8_t> block_is_allocated_;       // not using bool to avoid compacting
    std::vector<uint8_t> meta_block_is_allocated_;  // not using bool to avoid compacting

    // Metadata block indices that is not currently used (to reuse blocks instead of always allocating new ones)
    std::vector<size_t> free_meta_block_indices_;

    // Caches so most operations don't need to scan the entire free list. The allocated block table
    // will not rehash as I find the cost to not be worth it
    inline static constexpr size_t n_alloc_table_buckets = 512;          // Number of buckets in the hash table
    inline static constexpr size_t n_alloc_table_init_bucket_size = 10;  // Initial size of each bucket
    std::vector<std::vector<std::pair<DeviceAddr, size_t>>> allocated_block_table_;

    // Size segregated list of free blocks. Idea comes from the TLSF paper, but instead of aiming for realtime
    // the goal there is to not look at small blocks when allocating large blocks. Which the naive free list
    // algorithm does not do. Confiugring these 2 parameters is needs real world data, but for now it's just
    // number pulled out of thin air. Too low and it devolves into an array search, too high you pay cache misses

    // Size class index is calculated by taking the log2 of the block size divided by the base size
    // ex: size = 2048, base = 1024, log2(2048/1024) = 1, so size class index = 1
    inline static constexpr size_t size_segregated_base = 1024;  // in bytes
    const size_t size_segregated_count;                          // Number of size classes
    std::vector<std::vector<size_t>> free_blocks_segregated_by_size_;

    // internal functions
    // Given a block index, mark a chunk (from block start + offset to block start + offset + alloc_size) as allocated
    // Unused space is split into a new free block and retuened to the free list and the segregated list
    // NOTE: This function DOES NOT remove block_index from the segregated list. Caller should do that
    size_t allocate_in_block(size_t block_index, DeviceAddr alloc_size, size_t offset);

    inline size_t get_size_segregated_index(DeviceAddr size_bytes) const {
        // std::log2 is SLOW, so we use a simple log2 implementation for integers. I assume GCC compiles this to a
        // count leading zeros instruction then a subtraction.
        size_t lg = 0;
        size_t n = size_bytes / size_segregated_base;
        while (n >>= 1) {
            lg++;
        }
        return std::min(size_segregated_count - 1, lg);
    }
    // Put the block at block_index into the size segregated list at the appropriate index (data taken from
    // the SoA vectors)
    void insert_block_to_segregated_list(size_t block_index);

    // Allocate a new block and return the index to the block
    size_t alloc_meta_block(
        DeviceAddr address, DeviceAddr size, ssize_t prev_block, ssize_t next_block, bool is_allocated);
    // Free the block at block_index and mark it as free
    void free_meta_block(size_t block_index);

    // Operations on the allocated block table
    static size_t hash_device_address(DeviceAddr address);
    void insert_block_to_alloc_table(DeviceAddr address, size_t block_index);
    bool is_address_in_alloc_table(DeviceAddr address) const;
    std::optional<size_t> get_and_remove_from_alloc_table(DeviceAddr address);

    void update_lowest_occupied_address(DeviceAddr address);

    size_t find_free_block(DeviceAddr size, bool bottom_up);

    SearchPolicy policy_;
};

}  // namespace allocator
}  // namespace tt_metal
}  // namespace tt
