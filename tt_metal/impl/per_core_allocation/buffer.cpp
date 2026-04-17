// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

bool is_per_core_allocation(const Buffer& buffer) { return buffer.per_core_allocation_; }

DeviceAddr get_per_core_address(const Buffer& buffer, CoreCoord core) {
    TT_FATAL(
        buffer.per_core_allocation_, "get_per_core_address() called on buffer without per-core allocation enabled");
    auto it = buffer.per_core_addresses_.find(core);
    TT_FATAL(it != buffer.per_core_addresses_.end(), "No per-core address for core ({}, {})", core.x, core.y);
    return it->second;
}

const std::unordered_map<CoreCoord, DeviceAddr>& get_per_core_addresses(const Buffer& buffer) {
    return buffer.per_core_addresses_;
}

void copy_per_core_addresses(Buffer& dst, const Buffer& src) {
    TT_FATAL(
        dst.per_core_allocation_ && src.per_core_allocation_,
        "copy_per_core_addresses requires both buffers to use per-core allocation");
    dst.per_core_addresses_ = src.per_core_addresses_;
}

BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, bool enable) {
    args.per_core_allocation_ = enable;
    return args;
}

bool is_per_core_allocation(const BufferShardingArgs& args) { return args.per_core_allocation_; }

}  // namespace tt::tt_metal::experimental::per_core_allocation
