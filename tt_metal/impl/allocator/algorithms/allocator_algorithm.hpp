#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "hostdevcommon/common_values.hpp"

#include "tt_metal/impl/allocator/allocator_types.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

class Algorithm {
   public:
    Algorithm(u64 max_size_bytes, u64 offset_bytes, u64 min_allocation_size, u64 alignment)
        : max_size_bytes_(max_size_bytes), offset_bytes_(offset_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment), lowest_occupied_address_(std::nullopt) {
        log_assert(offset_bytes % this->alignment_ == 0, "Offset {} should be {} B aligned", offset_bytes, this->alignment_);
    }

    virtual ~Algorithm() {}

    u64 align(u64 address) const {
        u64 factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    u64 max_size_bytes() const { return max_size_bytes_; }

    std::optional<u64> lowest_occupied_address() const {
        if (not this->lowest_occupied_address_.has_value()) {
            return this->lowest_occupied_address_;
        }
        return this->lowest_occupied_address_.value() + this->offset_bytes_;
    }

    virtual void init() = 0;

    virtual std::vector<std::pair<u64, u64>> available_addresses(u64 size_bytes) const = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<u64> allocate(u64 size_bytes, bool bottom_up=true) = 0;

    virtual std::optional<u64> allocate_at_address(u64 absolute_start_address, u64 size_bytes) = 0;

    virtual void deallocate(u64 absolute_address) = 0;

    virtual void clear() = 0;

    virtual Statistics get_statistics() const = 0;

    virtual void dump_blocks(std::ofstream &out) const = 0;

   protected:
    u64 max_size_bytes_;
    u64 offset_bytes_;
    u64 min_allocation_size_;
    u64 alignment_;
    std::optional<u64> lowest_occupied_address_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
