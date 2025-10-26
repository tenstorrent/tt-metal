// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <hal_types.hpp>
#include <tt-metalium/allocator.hpp>

namespace tt {

namespace tt_metal {

namespace allocator {

class Algorithm {
public:
    Algorithm(
        DeviceAddr max_size_bytes, DeviceAddr offset_bytes, DeviceAddr min_allocation_size, DeviceAddr alignment) :
        max_size_bytes_(max_size_bytes),
        offset_bytes_(offset_bytes),
        min_allocation_size_(min_allocation_size),
        alignment_(alignment),
        lowest_occupied_address_(std::nullopt) {
        TT_ASSERT(
            offset_bytes % this->alignment_ == 0, "Offset {} should be {} B aligned", offset_bytes, this->alignment_);
    }

    virtual ~Algorithm() = default;

    DeviceAddr align(DeviceAddr address) const {
        DeviceAddr factor = (address + alignment_ - 1) / alignment_;
        return factor * alignment_;
    }

    DeviceAddr max_size_bytes() const { return max_size_bytes_; }

    std::optional<DeviceAddr> lowest_occupied_address() const {
        if (not this->lowest_occupied_address_.has_value()) {
            return this->lowest_occupied_address_;
        }
        return this->lowest_occupied_address_.value() + this->offset_bytes_;
    }

    virtual void init() = 0;

    virtual std::vector<std::pair<DeviceAddr, DeviceAddr>> available_addresses(DeviceAddr size_bytes) const = 0;

    // Returns all allocated address ranges as (start_address, end_address) pairs; start address is not sorted
    virtual std::vector<std::pair<DeviceAddr, DeviceAddr>> allocated_addresses() const = 0;

    // bottom_up=true indicates that allocation grows from address 0
    // Address limit is only used as a final check to see if selected address is > address limit
    // - Effectively, this means that address limit should only be used with top-down allocation
    virtual std::optional<DeviceAddr> allocate(
        DeviceAddr size_bytes, bool bottom_up = true, DeviceAddr address_limit = 0) = 0;

    virtual std::optional<DeviceAddr> allocate_at_address(DeviceAddr absolute_start_address, DeviceAddr size_bytes) = 0;

    virtual void deallocate(DeviceAddr absolute_address) = 0;

    virtual void clear() = 0;

    virtual Statistics get_statistics() const = 0;

    virtual void dump_blocks(std::ostream& out) const = 0;

    virtual MemoryBlockTable get_memory_block_table() const = 0;

    virtual void shrink_size(DeviceAddr shrink_size, bool bottom_up = true) = 0;

    virtual void reset_size() = 0;

protected:
    DeviceAddr max_size_bytes_;
    DeviceAddr offset_bytes_;
    DeviceAddr min_allocation_size_;
    DeviceAddr alignment_;
    DeviceAddr shrink_size_ = 0;
    std::optional<DeviceAddr> lowest_occupied_address_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
