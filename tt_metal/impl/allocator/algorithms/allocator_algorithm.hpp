#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "hostdevcommon/common_values.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

class Algorithm {
   public:
    Algorithm(u32 max_size_bytes, u32 min_allocation_size, u32 alignment)
        : max_size_bytes_(max_size_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment) {}

    virtual ~Algorithm() {}

    u32 align(u32 address) const {
        u32 factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    u32 max_size_bytes() const { return max_size_bytes_; }

    virtual void init() = 0;

    virtual std::vector<std::pair<u32, u32>> available_addresses(u32 size_bytes) const = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<u32> allocate(u32 size_bytes, bool bottom_up=true) = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<u32> allocate_at_address(u32 start_address, u32 size_bytes) = 0;

    virtual void deallocate(u32 address) = 0;

    virtual void clear() = 0;

   protected:
    u32 max_size_bytes_;
    u32 min_allocation_size_;
    u32 alignment_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
