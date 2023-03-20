#pragma once

#include "tt_metal/impl/memory_manager/red_black_tree.hpp"
#include <cstdint>
#include <vector>

namespace tt {

namespace tt_metal {

class MemoryManager {
   public:
    struct Block {
        uint32_t address;
        uint32_t size;
        Block *prev = nullptr;
        Block *next = nullptr;
    };

    MemoryManager(uint32_t max_size_bytes);

    uint32_t malloc(uint32_t size_bytes);

    uint32_t reserve(uint32_t start_address, uint32_t size_bytes);

    void free(uint32_t address);

    uint32_t peak() const;

    void clear();

   private:
    uint32_t reserve_free_space(uint32_t size_bytes);

    uint32_t get_address(uint32_t allocated_size);

    void insert_block(Block *block);

    uint32_t coalesce(Block *block_to_free);

    uint32_t max_size_bytes_;
    uint32_t min_allocation_size_;
    red_black_tree free_tree_;
    Block *head_;
    Block *tail_;
};

}  // namespace tt_metal

}  // namespace tt
