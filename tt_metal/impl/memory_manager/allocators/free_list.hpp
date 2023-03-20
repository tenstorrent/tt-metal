#pragma once

#include "tt_metal/impl/memory_manager/allocators/allocator.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

class FreeList : public Allocator {
   public:
    enum class SearchPolicy {
        BEST = 0,
        FIRST = 1
    };

    FreeList(uint32_t max_size_bytes, uint32_t min_allocation_size, SearchPolicy search_policy);

    ~FreeList();

    void init();

    std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t size_bytes) const;

    uint32_t allocate(uint32_t size_bytes);

    uint32_t reserve(uint32_t start_address, uint32_t size_bytes);

    void deallocate(uint32_t address);

    void clear();

   private:
    struct Block {
        uint32_t address;
        uint32_t size;
        Block *prev_block = nullptr;
        Block *next_block = nullptr;
        Block *prev_free = nullptr;
        Block *next_free = nullptr;
    };

    void dump_blocks() const;

    bool is_allocated(Block *block) const;

    Block *search_best(uint32_t size_bytes);

    Block *search_first(uint32_t size_bytes);

    Block *search(uint32_t size_bytes);

    void split_free_block(Block *to_be_allocated, uint32_t size_bytes);

    void segment_free_block(Block *to_be_split, uint32_t address, uint32_t size_bytes);

    void allocate_free_block(Block *to_be_allocated, uint32_t size_bytes);

    Block *find_block(uint32_t address);

    void reset();

    SearchPolicy search_policy_;
    Block *block_head_;
    Block *free_block_head_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
