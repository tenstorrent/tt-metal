// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffers/simulator_direct_write.hpp"

#if defined(TT_UMD_BUILD_SIMULATION)

#include "llrt/rtoptions.hpp"
#include <tt-metalium/experimental/core_subset_write/buffer_write.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::tt_sim {

bool is_direct_write_enabled(const DirectWriteGuard& guard, const void* src, const BufferRegion& region) {
    if (guard.target != tt::TargetDevice::Simulator) {
        return false;
    }
    if (!guard.cq_idle) {
        return false;
    }
    if (guard.rtoptions == nullptr || !guard.rtoptions->get_simulator_direct_tensor_writes()) {
        return false;
    }
    if (src == nullptr) {
        return false;
    }
    if (region.offset != 0) {
        return false;
    }
    return true;
}

void write_shard(
    Buffer& shard_view, const void* src, const BufferRegion& region, const CoreRangeSet* logical_core_filter) {
    auto payload = ttsl::Span<const uint8_t>(static_cast<const uint8_t*>(src), static_cast<size_t>(region.size));
    if (logical_core_filter != nullptr) {
        experimental::core_subset_write::WriteToBuffer(shard_view, payload, *logical_core_filter);
    } else {
        detail::WriteToBuffer(shard_view, payload);
    }
}

bool try_direct_write(
    const DirectWriteGuard& guard,
    Buffer& shard_view,
    const void* src,
    const BufferRegion& region,
    const CoreRangeSet* logical_core_filter) {
    if (!is_direct_write_enabled(guard, src, region)) {
        return false;
    }
    write_shard(shard_view, src, region, logical_core_filter);
    return true;
}

}  // namespace tt::tt_metal::tt_sim

#endif  // defined(TT_UMD_BUILD_SIMULATION)
