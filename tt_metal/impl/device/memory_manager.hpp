#pragma once

#include <cstdint>

namespace tt {

namespace tt_metal {

// Simple stack allocator
class MemoryManager {
   public:
    MemoryManager(uint32_t max_size_bytes) :
        max_size_bytes_(max_size_bytes), start_ptr_(0), offset_(0) {}

    uint32_t malloc(uint32_t size_bytes);

    void free(uint32_t address);

    uint32_t reserve(uint32_t start_address, uint32_t size_bytes);

    uint32_t peak() const { return offset_; }

    void clear();

   private:
    uint32_t max_size_bytes_;
    uint32_t start_ptr_;
    uint32_t offset_;
};

}  // namespace tt_metal

}  // namespace tt
