// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

namespace tt::tt_metal {

// Host timebase for the real-time profiler's device<->host clock sync and the mapping exposed to callbacks
// (ProgramRealtimeClockSync). CLOCK_MONOTONIC via std::chrono::steady_clock, so a device timestamp mapped through the
// published clock_sync is directly comparable to a consumer's own steady_clock::now() — on any thread, since
// CLOCK_MONOTONIC is system-wide. The Tracy consumer converts this to Tracy's rdtsc timebase itself (see
// realtime_profiler_tracy_consumer.cpp); nothing else needs rdtsc.
inline int64_t realtime_profiler_host_timestamp() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace tt::tt_metal
