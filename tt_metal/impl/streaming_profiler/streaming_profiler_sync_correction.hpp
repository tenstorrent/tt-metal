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
// newest node along `tangent` (d value / d at) as far as the series' cover reaches. sigma_ns is the standard
// deviation of `value` at the node, in ns.
template <typename Key>
struct PlacementNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
    float sigma_ns = 0.0f;
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

// Append-only series of frozen nodes, strictly increasing in their key: one per chip, written by the d2d sync
// consumer, and the host series, written by the probe; read by every consumer thread converting a record's time.
// Nodes never move within a capture and a series keeps its newest kSeriesNodes, so a reader keeps a thread-local
// cursor on the segment it last used and converts without touching shared state until the record leaves the segment.
// Before a chip's first node, or the host's, its records have no place on the host timeline; before the oldest kept
// node they convert on its tangent.
//
// A series' cover is the key up to which the newest node's tangent has been confirmed: a record at or before it
// converts against frozen data on both sides. Writer order is nodes, count, cover (release); readers load the cover
// before the count (acquire), so a cover a reader sees implies the nodes behind it.
class SyncCorrections {
public:
    static constexpr uint32_t kMaxChips = 256;
    // Nodes a series keeps (32 MB); the oldest go as newer ones arrive. Nodes come per local clock step and per host
    // burst, so this spans hours of a capture and any consumer's lag behind the sync.
    static constexpr uint32_t kSeriesNodes = 1u << 20;
    // Appends a node past every earlier one (a node at the last node's key is dropped) and moves the cover to it.
    static void append(uint32_t chip_id, SyncNode node);
    // The newest node's tangent holds up to cover_ticks; the cover never moves back.
    static void extend(uint32_t chip_id, int64_t cover_ticks);
    // The series is complete for the capture: every later instant converts on the newest tangent.
    static void finish(uint32_t chip_id);
    // Empties a chip's series for a new capture.
    static void clear(uint32_t chip_id);
    static void append_host(HostNode node);
    static void extend_host(double cover_root);
    // The root refclk tick of a chip's eth wall tick; 0 before the chip's first node.
    static double lookup_root(uint32_t chip_id, int64_t wall) noexcept;
    // The host TSC tick of a chip's eth wall tick; 0 before the chip's first node or the host's.
    static int64_t lookup_tsc(uint32_t chip_id, int64_t wall) noexcept;
    // The uncertainty of that placement against other chips' records: kSigmas standard deviations of its segment's
    // nodes plus the fleet's path asymmetry; INT64_MAX before the chip's first node.
    static int64_t lookup_error_ns(uint32_t chip_id, int64_t wall) noexcept;
    // Wall tick `wall` of chip `chip_id` on host_clock (tenths of a ns of the TSC): the chip series and the host
    // series composed into one line per segment pair, one multiply-add per record while a batch stays inside it. 0
    // when nothing places the tick yet.
    static int64_t place_host(uint32_t chip_id, int64_t wall) noexcept;
    static constexpr double kSigmas = 3.0;
    // The largest loop closure the link solutions have shown, the part of a placement's error the loops can see
    // but no link's stamps can.
    static void set_asymmetry_ns(double ns) noexcept;
    // How many nodes a chip has (0 = none).
    static size_t published(uint32_t chip_id) noexcept;
    static size_t host_published() noexcept;
    // A copy of the host series, for the capture-end dumps.
    static std::vector<HostNode> host_nodes();
    // The wall tick the chip's series covers: INT64_MIN before its first node, INT64_MAX once finished.
    static int64_t cover_ticks(uint32_t chip_id) noexcept;
    // Moves whenever any chip's cover does, so a consumer holding batches re-reads covers only then.
    static uint64_t cover_generation() noexcept;
    // The steady_clock view of the host TSC, kept by the host probe; readers cache it per thread.
    static void set_steady(const SteadySegment& segment) noexcept;
    static int64_t tsc_to_mono_ns(int64_t tsc) noexcept;
};

// A named (host TSC tick, value) series a consumer computes once a capture is complete -- the d2d sync's running
// cross-chip rate estimates -- for a plotting sink to place on the device timeline. Not a hot path: published once at
// capture end, drained once by the sink.
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
