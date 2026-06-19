// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include "llrt/tt_target_device.hpp"

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal::tt_sim {

// Inputs required to evaluate the tt-sim direct H2D write guard for fast-dispatch mesh uploads.
// cq_idle is owned by the mesh command queue; try_direct_write may still allow a narrow
// simulator-only sharded-L1 staging write when ordered FD work is pending.
struct DirectWriteGuard {
    tt::TargetDevice target = tt::TargetDevice::Invalid;
    bool cq_idle = false;
    const tt::llrt::RunTimeOptions* rtoptions = nullptr;
};

#if defined(TT_UMD_BUILD_SIMULATION)

// True when tt-sim may use the ordered direct-write path for this shard write.
bool is_direct_write_enabled(const DirectWriteGuard& guard, const void* src, const BufferRegion& region);

// True when this shard can use simulator direct writes after the caller has
// satisfied command queue ordering.
bool is_direct_write_candidate(
    const DirectWriteGuard& guard, Buffer& shard_view, const void* src, const BufferRegion& region);

// Synchronous host-to-device shard write used by the tt-sim fast path.
void write_shard(
    Buffer& shard_view, const void* src, const BufferRegion& region, const CoreRangeSet* logical_core_filter);

// Returns true when the tt-sim guard fired and the write completed synchronously.
bool try_direct_write(
    const DirectWriteGuard& guard,
    Buffer& shard_view,
    const void* src,
    const BufferRegion& region,
    const CoreRangeSet* logical_core_filter);

#else

inline bool try_direct_write(
    const DirectWriteGuard& /*guard*/,
    Buffer& /*shard_view*/,
    const void* /*src*/,
    const BufferRegion& /*region*/,
    const CoreRangeSet* /*logical_core_filter*/) {
    return false;
}

#endif

}  // namespace tt::tt_metal::tt_sim
