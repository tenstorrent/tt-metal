#pragma once

#include <cstdint>
#include <vector>

namespace tt {

namespace tt_metal {

class MemoryManager {
   public:
    MemoryManager(uint32_t max_size_bytes);

    uint32_t malloc(uint32_t size_bytes);

    uint32_t reserve(uint32_t start_address, uint32_t size_bytes);

    void free(uint32_t address, uint32_t size_bytes);

    uint32_t peak() const;

    void clear();

   private:
    bool ancestor_in_use(uint32_t child_level, uint32_t child_index);

    uint32_t level_of_used_ancestor(uint32_t child_level, uint32_t child_index);

    void mark_ancestors_as_split(uint32_t child_level, uint32_t child_index);

    uint32_t max_size_bytes_;
    uint32_t num_levels_;
    uint32_t num_blocks_;
    uint32_t min_allocation_size_;
    std::vector<uint8_t> in_use_;
    std::vector<uint8_t> is_split_;
};

}  // namespace tt_metal

}  // namespace tt
