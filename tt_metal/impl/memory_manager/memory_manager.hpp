#pragma once

#include "tt_metal/impl/memory_manager/allocators/free_list.hpp"
#include <cstdint>
#include <vector>

namespace tt {

namespace tt_metal {

class MemoryManager {
   public:
    MemoryManager(uint32_t max_size_bytes);

    ~MemoryManager();

    uint32_t allocate(uint32_t size_bytes);

    uint32_t reserve(uint32_t start_address, uint32_t size_bytes);

    void deallocate(uint32_t address);

    std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t size_bytes) const;

    void clear();

   private:
    allocator::Allocator *allocator_;

};

}  // namespace tt_metal

}  // namespace tt
