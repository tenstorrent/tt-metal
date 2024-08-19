// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    Algorithm(uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment)
        : max_size_bytes_(max_size_bytes), offset_bytes_(offset_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment), lowest_occupied_address_(std::nullopt) {
        TT_ASSERT(offset_bytes % this->alignment_ == 0, "Offset {} should be {} B aligned", offset_bytes, this->alignment_);
    }

    virtual ~Algorithm() {}

    uint64_t align(uint64_t address) const {
        uint64_t factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    uint64_t max_size_bytes() const { return max_size_bytes_; }

    std::optional<uint64_t> lowest_occupied_address() const {
        if (not this->lowest_occupied_address_.has_value()) {
            return this->lowest_occupied_address_;
        }
        return this->lowest_occupied_address_.value() + this->offset_bytes_;
    }

    virtual void init() = 0;

    virtual std::vector<std::pair<uint64_t, uint64_t>> available_addresses(uint64_t size_bytes) const = 0;

    // bottom_up=true indicates that allocation grows from address 0
    virtual std::optional<uint64_t> allocate(uint64_t size_bytes, bool bottom_up=true, uint64_t address_limit=0) = 0;

    virtual std::optional<uint64_t> allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes) = 0;

    virtual void deallocate(uint64_t absolute_address) = 0;

    virtual void clear() = 0;

    virtual Statistics get_statistics() const = 0;

    virtual void dump_blocks(std::ofstream &out) const = 0;

   protected:
    uint64_t max_size_bytes_;
    uint64_t offset_bytes_;
    uint64_t min_allocation_size_;
    uint64_t alignment_;
    std::optional<uint64_t> lowest_occupied_address_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
