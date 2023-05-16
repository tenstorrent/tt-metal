#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace tt {

namespace tt_metal {

namespace allocator {

class Algorithm {
   public:
    Algorithm(uint32_t max_size_bytes, uint32_t min_allocation_size, uint32_t alignment)
        : max_size_bytes_(max_size_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment) {}

    virtual ~Algorithm() {}

    uint32_t align(uint32_t address) const {
        uint32_t factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    uint32_t max_size_bytes() const { return max_size_bytes_; }

    virtual void init() = 0;

    virtual std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t size_bytes) const = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<uint32_t> allocate(uint32_t size_bytes, bool bottom_up=true) = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<uint32_t> allocate_at_address(uint32_t start_address, uint32_t size_bytes) = 0;

    virtual void deallocate(uint32_t address) = 0;

    virtual void clear() = 0;

   protected:
    uint32_t max_size_bytes_;
    uint32_t min_allocation_size_;
    uint32_t alignment_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
