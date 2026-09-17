// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

// One frozen node of a placement series: at `at` the placement is `value`, linear to the next node, and past the
// newest node along `tangent` (d value / d at) as far as the series' cover reaches.
template <typename Key>
struct PlacementNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
// A chip's series: its eth wall tick -> the root chip's refclk tick, from the link solutions and the local fits alone,
// so two chips' records at one instant differ by nothing the host contributes. Worker lanes reach the eth wall
// domain through their chip's constant tile offset, so one series places every record of a chip.
using SyncNode = PlacementNode<int64_t>;
// The fleet's one host series: the root's refclk tick -> host TSC tick, from the host probe.
using HostNode = PlacementNode<double>;

// The host TSC on CLOCK_MONOTONIC, one line between two NTP slews: mono_ns = mono0 + (tsc - tsc0) * ns_per_tick.
struct SteadySegment {
    int64_t tsc0 = 0, mono0 = 0;
    double ns_per_tick = 0.0;
    bool ok = false;
    int64_t mono_of(int64_t tsc) const { return mono0 + std::llrint(static_cast<double>(tsc - tsc0) * ns_per_tick); }
};

// The sync engine's placement map: one series per chip (its eth wall tick -> the root chip's refclk tick) and the
// host series (the root's refclk tick -> host TSC tick). The sync engine writes the chip series and the host probe
// the host series; the service's consumer threads read them to place records. Each series is append-only within a
// capture, strictly increasing in its key, and keeps its newest kSeriesNodes: a record before the oldest kept node
// converts on that node's tangent. Before a chip's first node, or the host's, its records have no place on the host
// timeline.
//
// Reads never lock and never block the writer (IndexedRing). A reader keeps a thread-local cursor on the segment it
// last converted in and converts without touching shared state until a record leaves the segment. A series' cover
// is the key up to which the newest node's tangent has been confirmed: a record at or before it converts against
// frozen data on both sides.
class PlacementMap {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Nodes a series keeps (32 MB at most); the oldest go as newer ones arrive. Nodes come per local clock step and
    // per host burst, so this spans hours of a capture and any consumer's lag behind the sync.
    static constexpr uint32_t kSeriesNodes = 1u << 20;

    PlacementMap();
    ~PlacementMap();
    PlacementMap(const PlacementMap&) = delete;
    PlacementMap& operator=(const PlacementMap&) = delete;

    // Appends a node past every earlier one (a node at the last node's key is dropped) and moves the cover to it.
    void append(uint32_t chip_id, SyncNode node);
    // The newest node's tangent holds up to cover_ticks; the cover never moves back.
    void extend(uint32_t chip_id, int64_t cover_ticks);
    // The series is complete for the capture: every later instant converts on the newest tangent.
    void finish(uint32_t chip_id);
    // Empties a chip's series for a new capture.
    void clear(uint32_t chip_id);
    void append_host(HostNode node);

    // The root refclk tick of a chip's eth wall tick; 0 before the chip's first node.
    double lookup_root(uint32_t chip_id, int64_t wall) const noexcept;
    // The host TSC tick of a chip's eth wall tick; 0 before the chip's first node or the host's.
    int64_t lookup_tsc(uint32_t chip_id, int64_t wall) const noexcept;
    // Wall tick `wall` of chip `chip_id` on host_clock (tenths of a ns of the TSC): the chip series and the host
    // series composed into one line per segment pair, one multiply-add per record while a batch stays inside it. 0
    // when nothing places the tick yet.
    int64_t place_host(uint32_t chip_id, int64_t wall) const noexcept;
    // The host TSC tick of a root refclk tick; 0 before the host's first node.
    double host_tsc(double root) const noexcept;
    size_t host_published() const noexcept;
    // The wall tick the chip's series covers: INT64_MIN before its first node, INT64_MAX once finished.
    int64_t cover_ticks(uint32_t chip_id) const noexcept;
    // Moves whenever any chip's cover does, so a consumer holding batches re-reads covers only then.
    uint64_t cover_generation() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The host TSC on steady_clock as the host probe measures it: one segment for the process, readable from any thread
// and cached per thread. No capture is involved, so the API's steady_time() reads it with no device open.
class SteadyView {
public:
    static void set(const SteadySegment& segment) noexcept;
    // A TSC/CLOCK_MONOTONIC pair taken here stands in until a probe publishes a segment.
    static int64_t mono_ns(int64_t tsc) noexcept;
};

// A named (host TSC tick, value) series a consumer computes once a capture is complete -- the d2d sync's error per
// link and each chip's AICLK -- for a plotting sink to place on the device timeline. Not a hot path: published once
// at capture end, drained once by the sink.
struct SyncPlotPoint {
    int64_t tsc = 0;
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
