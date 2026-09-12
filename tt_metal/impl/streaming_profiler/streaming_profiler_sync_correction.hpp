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

// One frozen node of a chip's time-indexed correction, keyed by the record's host time BEFORE the correction (its
// static anchor's placing): the correction is delta_ns there, linear to the next node, and past the newest node it
// follows that node's tangent (the correction's slope along the chip's current constant-rate run) as far as the
// series' cover reaches. The d2d sync freezes a node where the map bends (a DVFS transition) and where its estimate
// has drifted from the frozen tangent; between nodes the chip's clock ran at one rate.
struct SyncNode {
    int64_t host_ns = 0;
    double delta_ns = 0.0;
    double tangent = 0.0;
    float sigma_ns = 0.0f;  // standard deviation of delta_ns
};

enum class SyncSeries : uint8_t { Linked, Local };

// Per-chip append-only series of frozen nodes, strictly increasing in host_ns, written by the d2d sync consumer and
// read by every consumer thread converting a record's time. Nodes never move or go away within a capture, so a
// reader keeps a thread-local cursor on the segment it last used and converts without touching shared state until
// the record leaves the segment. Before any node the correction is 0 and records convert exactly as they did
// without the d2d sync; before a chip's first node it is that node's value.
//
// The cover is the base host time up to which the newest node's tangent has been confirmed: a record at or before it
// converts against frozen data on both sides. Writer order is nodes, count, cover (release); readers load the cover
// before the count (acquire), so a cover a reader sees implies the nodes behind it.
class SyncCorrections {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Beyond the cover the newest tangent is extended, but only this far; further out the correction holds constant
    // rather than extrapolating a slope.
    static constexpr int64_t kHoldNs = 50'000'000;
    // Appends a node past every earlier one (a node at the last node's ns is dropped) and moves the cover to it.
    static void append(uint32_t chip_id, SyncNode node, SyncSeries series = SyncSeries::Linked);
    // The newest node's tangent holds up to cover_ns; the cover never moves back.
    static void extend(uint32_t chip_id, int64_t cover_ns, SyncSeries series = SyncSeries::Linked);
    // The series is complete for the capture: every later instant converts on the newest tangent.
    static void finish(uint32_t chip_id, SyncSeries series = SyncSeries::Linked);
    // Empties both of a chip's series for a new capture.
    static void clear(uint32_t chip_id);
    static int64_t lookup_ns(uint32_t chip_id, int64_t host_ns, SyncSeries series = SyncSeries::Linked) noexcept;
    // Both ends of one record; the end never precedes the start.
    static void lookup_span_ns(
        uint32_t chip_id, int64_t start_ns, int64_t end_ns, int64_t& d_start, int64_t& d_end) noexcept;
    // The uncertainty of a record's corrected host time against other chips' records: kSigmas standard deviations
    // of its segment's nodes plus the fleet's path asymmetry; INT64_MAX before the chip's first node.
    static int64_t lookup_error_ns(uint32_t chip_id, int64_t host_ns) noexcept;
    static constexpr double kSigmas = 3.0;
    // The largest loop closure the link solutions have shown, the part of a placement's error the loops can see
    // but no link's stamps can.
    static void set_asymmetry_ns(double ns) noexcept;
    // How many linked nodes a chip has (0 = none).
    static size_t published(uint32_t chip_id) noexcept;
    // The base host time the chip's linked series covers: INT64_MIN before its first node, INT64_MAX once finished.
    static int64_t cover_ns(uint32_t chip_id) noexcept;
    // Moves whenever any chip's linked cover does, so a consumer holding batches re-reads covers only then.
    static uint64_t cover_generation() noexcept;
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
