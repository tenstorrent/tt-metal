#pragma once

#include <cstdint>
#include <vector>

namespace tt {

namespace tt_metal {

namespace allocator {

class Allocator {
   public:
    Allocator(uint32_t max_size_bytes, uint32_t min_allocation_size)
        : max_size_bytes_(max_size_bytes), min_allocation_size_(min_allocation_size) {}

    virtual ~Allocator() {}

    virtual void init() = 0;

    virtual std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t size_bytes) const = 0;

    virtual uint32_t allocate(uint32_t size_bytes) = 0;

    virtual uint32_t reserve(uint32_t start_address, uint32_t size_bytes) = 0;

    virtual void deallocate(uint32_t address) = 0;

    virtual void clear() = 0;

   protected:

    uint32_t max_size_bytes_;
    uint32_t min_allocation_size_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
