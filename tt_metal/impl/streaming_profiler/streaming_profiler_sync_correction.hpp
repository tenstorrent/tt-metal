// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

// One piece of a chip's time-indexed correction to its baked host-time anchor: over wall ticks [tick_lo, tick_hi]
// the correction is delta_ns_lo + slope_ns_per_tick * (ticks - tick_lo). The d2d sync publishes one per 1 ms
// tracker bucket -- the AICLK that actually applied over that millisecond, and for a non-root chip the link onto the
// root's timeline -- so a Record's host time is its static anchor composed with this term.
struct SyncSegment {
    uint64_t tick_lo = 0, tick_hi = 0;
    double delta_ns_lo = 0.0;
    double slope_ns_per_tick = 0.0;
};

// Per-chip published series. Readers (every consumer thread converting a record's time) are lock-free: they load the
// current snapshot and binary-search it; a publish swaps in a new snapshot. Before any publish, or before a chip's
// first segment, the correction is 0 and records convert exactly as they did without the d2d sync.
class SyncCorrections {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Beyond the last segment a live sink slightly ahead of the fit extends the last line, but only this far in
    // ticks (~50 ms at 1.35 GHz); further out the correction holds constant rather than extrapolating a slope.
    static constexpr uint64_t kHoldTicks = 67'500'000;

    static void publish(uint32_t chip_id, std::vector<SyncSegment> segments);
    static void clear(uint32_t chip_id);
    static int64_t lookup_ns(uint32_t chip_id, uint64_t ticks) noexcept;
    // A parallel LOCAL-only series (each chip's own-anchor + local-AICLK term, no cross-chip link). Used only
    // for the Tracy/CSV local-vs-linked plots; Record::host_time uses the linked series above.
    static void publish_local(uint32_t chip_id, std::vector<SyncSegment> segments);
    static int64_t lookup_local_ns(uint32_t chip_id, uint64_t ticks) noexcept;
    // How many segments a chip currently has published (0 = none).
    static size_t published(uint32_t chip_id) noexcept;
};

}  // namespace tt::tt_metal::streaming_profiler
