// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/hal_types.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental {

// A region of continous memory in a device bank
struct DeviceSpan {
    using address_type = DeviceAddr;
    using size_type = uint32_t;

    // Default constructed device span is an empty span.
    constexpr DeviceSpan() : addr_(0), size_(0) {}

    // Construct device from a address and size
    constexpr DeviceSpan(address_type addr, size_type size) : addr_(addr), size_(size) {
        TT_FATAL(addr != 0, "Constructing a DeviceSpan from 0 address");
    }

    [[nodiscard]] constexpr address_type base_address() const { return addr_; }

    [[nodiscard]] constexpr size_type size() const { return size_; }

    [[nodiscard]] constexpr bool empty() const { return size_ == 0; }

    [[nodiscard]] constexpr DeviceSpan subspan(size_type offset, size_type size) const {
        // TODO: bound check
        return DeviceSpan(addr_ + offset, size);
    }

private:
    DeviceAddr addr_;
    uint32_t size_;
};

};  // namespace tt::tt_metal::experimental
