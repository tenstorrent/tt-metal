// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

// One node of a chip's time-indexed correction, keyed by the record's host time BEFORE the correction (its static
// anchor's placing): the correction is delta_ns there and linear between neighbouring nodes. The d2d sync places a
// node at every DVFS transition and at each publish, so between nodes the chip's clock ran at one rate.
struct SyncNode {
    int64_t host_ns = 0;
    double delta_ns = 0.0;
};

// Per-chip published series of nodes, strictly increasing in host_ns. Readers (every consumer thread converting a
// record's time) are lock-free: they load the current snapshot and binary-search it; a publish swaps in a new
// snapshot. Before any publish the correction is 0 and records convert exactly as they did without the d2d sync;
// before a chip's first node it is that node's value.
class SyncCorrections {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Beyond the last node a live sink slightly ahead of the fit extends the last two nodes' line, but only this far;
    // further out the correction holds constant rather than extrapolating a slope.
    static constexpr int64_t kHoldNs = 50'000'000;

    // `nodes` must be strictly increasing in host_ns.
    static void publish(uint32_t chip_id, std::vector<SyncNode> nodes);
    static void clear(uint32_t chip_id);
    static int64_t lookup_ns(uint32_t chip_id, int64_t host_ns) noexcept;
    // Both ends of one record from the same snapshot; the end never precedes the start.
    static void lookup_span_ns(
        uint32_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept;
    // A parallel LOCAL-only series (each chip's own-anchor + local-AICLK term, no cross-chip link). Used only
    // for the Tracy/CSV local-vs-linked plots; Record::host_time uses the linked series above.
    static void publish_local(uint32_t chip_id, std::vector<SyncNode> nodes);
    static int64_t lookup_local_ns(uint32_t chip_id, int64_t host_ns) noexcept;
    // How many nodes a chip currently has published (0 = none).
    static size_t published(uint32_t chip_id) noexcept;
    // The host time of the chip's last published linked node; INT64_MIN before any publish. A record behind it
    // converts against nodes on both sides; ahead of it, against the last two nodes' line extended.
    static int64_t published_until_ns(uint32_t chip_id) noexcept;
};

// A named (host ns, value) series a consumer computes once a capture is complete -- the d2d sync's running
// cross-chip rate estimates -- for a plotting sink to place on the device timeline. Not a hot path: published once at
// capture end, drained once by the sink.
struct SyncPlotPoint {
    int64_t host_ns = 0;
    double value = 0.0;
};
class SyncPlots {
public:
    static void publish(std::string name, std::vector<SyncPlotPoint> points);
    static std::vector<std::pair<std::string, std::vector<SyncPlotPoint>>> drain();
    // The computing consumer and the draining sink run on their own threads: the consumer declares its series
    // pending at attach and complete after its final publish, and a sink drains only once they are complete (or the
    // wait runs out).
    static void expect();
    static void complete();
    static void wait_complete(std::chrono::milliseconds timeout);
};

}  // namespace tt::tt_metal::streaming_profiler
