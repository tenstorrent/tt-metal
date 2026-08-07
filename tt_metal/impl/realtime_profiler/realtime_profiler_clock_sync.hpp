// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

// Maps device tick timestamps onto host time by interpolating between retained clock probes. Pure arithmetic over
// what it is fed: probes come from whoever reads the device clock -- RealtimeProfilerClockSync in production, tests
// directly -- and nothing here touches a device.
//
// Single-threaded: retain() and map_record() belong to the thread draining the device.
class RealtimeProfilerClockMapping {
public:
    // A probe, placed at the midpoint of the bracket its read fell in. Two of them map any device timestamp between
    // them via their secant, whatever the clock did in between.
    struct Anchor {
        std::chrono::steady_clock::time_point host;
        uint64_t ticks = 0;
        std::chrono::nanoseconds bracket{};
    };

    // What one record publishes with, covering both of its timestamps.
    //
    // One timestamp is anchored exactly where its bracketing probes place it; the other is derived through the
    // published rate. sync_error covers both: the anchoring reads' brackets, the measured departure of the clock from
    // their secant, and the derived timestamp's exposure -- its distance from the anchor times the disagreement
    // between the published rate and the pair's own. What it cannot cover is a rate step inside the bracketing pair
    // that no probe has resolved yet; the didt suite bounds what that costs in practice.
    struct RecordMapping {
        experimental::ProgramRealtimeClockSync clock_sync;
        double frequency = 0.0;
    };

    // How wide a window the published rate is measured across. Wide is what keeps the rate quiet -- a chord's own
    // slope is uncertain by (both brackets)/span, thousands of ppm at chord width, and every duration a consumer
    // computes divides by the published rate -- while wider still would smooth over the ~5200ppm AICLK steps the rate
    // has to keep tracking.
    static constexpr auto kRateBaseline = std::chrono::milliseconds(4);

    // Overwrites the oldest probe when full. The pair around an undecoded record's end can never be overwritten: the
    // receiver probes once per drained batch, and on the idle floor only when nothing is in flight, so a record sees
    // a few dozen probes at most between ending and being decoded (asserted against the FIFO geometry in the
    // receiver). Only a record's start can predate the ring -- a program that ran longer than the ring spans -- and
    // that start is derived from the record's end instead of a pair of its own.
    static constexpr size_t kProbeHistoryCapacity = 4096;

    // Retains `probe`. One that does not advance both clocks past the newest retained probe is dropped -- a real
    // counter and steady_clock cannot produce one -- so the ring is strictly monotone in host and ticks and no reader
    // below has to check a pair's orientation.
    void retain(const Anchor& probe);

    // What to publish a record with. Nullopt only when no retained probe precedes either timestamp, which after
    // warm-up cannot happen to a real record; the caller treats it as corruption and rejects the page.
    //
    // Records arrive in tick order, so the probe pair one is placed between also covers the next several; that pair
    // is held here and refreshed when a record's start passes it.
    [[nodiscard]] std::optional<RecordMapping> map_record(uint64_t start_ticks, uint64_t end_ticks);

    // sync_error of the last record mapped, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_sync_error() const { return last_sync_error_; }

private:
    // The probe pair around a run of records, resolved once and reused until a record's start passes `close_ticks`.
    // The slope is inverted here because the alternative is a division per record on the drain thread.
    struct Chord {
        std::chrono::nanoseconds sync_error{};
        // Measured across kRateBaseline where the history allows, this pair's own slope until then.
        double frequency = 0.0;
        uint64_t open_ticks = 0;
        double open_host_ns = 0.0;
        double inv_chord_rate = 0.0;
        uint64_t close_ticks = 0;
    };

    // The tightest pair of retained probes around `ticks`, as a chord. Nullopt only when the ring holds nothing
    // before `ticks`. Total otherwise, so a caller that has just probed always gets an answer and nothing is held
    // back for a later pass -- a refusal here once stalled a device's whole data path, being re-asked about the same
    // record with inputs that never changed.
    [[nodiscard]] std::optional<Chord> chord_around(uint64_t ticks) const;

    [[nodiscard]] static double host_ns_on(const Chord& chord, uint64_t ticks) {
        return chord.open_host_ns +
               (static_cast<double>(ticks) - static_cast<double>(chord.open_ticks)) * chord.inv_chord_rate;
    }

    // The mapping that restates one timestamp's placement in terms of `frequency`, so that timestamp lands exactly
    // where it was placed whatever rate is published with it.
    [[nodiscard]] static RecordMapping anchored(
        double frequency, uint64_t anchor_ticks, double anchor_host_ns, std::chrono::nanoseconds error);

    // What a timestamp `dur_ticks` away from the anchor is charged for riding the published rate: its distance times
    // the disagreement between that rate and the pair's own.
    [[nodiscard]] static std::chrono::nanoseconds rate_exposure(const Chord& chord, uint64_t dur_ticks);

    // The endpoint term of an interpolated timestamp's error: it lands on the secant through two measured points, so
    // it inherits how well those points are placed. A clock that moved within the interval adds to this; see
    // measured_bow.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    // How far `interior` lies off the chord through `open` and `close`, less its own read noise. A probe inside a
    // chord was not fitted to it, so this is the clock's departure at that point -- measured, where the alternative is
    // to infer it from how much two chords' slopes differ and hope the difference was the clock rather than the reads.
    [[nodiscard]] static std::chrono::nanoseconds departure_from_chord(
        const Anchor& open, const Anchor& close, const Anchor& interior);

    // What the retained probes say the clock did between `close_index - 1` and `close_index` that the line through
    // them does not capture. Zero when there is no third probe to read it from, which is the absence of evidence
    // rather than a claim of linearity.
    [[nodiscard]] std::chrono::nanoseconds measured_bow(uint64_t close_index) const;

    // The rate across the retained probe history, or nullopt while it is too narrow to beat a single chord.
    [[nodiscard]] std::optional<double> baseline_rate() const;

    // Index of the oldest retained probe whose counter read reached `ticks`, or probes_end_ when none has. Probes are
    // appended in tick order, so this bisects: the retained span grows with the backlog, and scanning it per record
    // is what turns a backlog into a stall.
    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t ticks) const;

    [[nodiscard]] uint64_t oldest_probe() const {
        return probes_end_ > kProbeHistoryCapacity ? probes_end_ - kProbeHistoryCapacity : 0;
    }
    [[nodiscard]] const Anchor& probe_at(uint64_t index) const { return probe_history_[index % kProbeHistoryCapacity]; }

    // Overwrites its oldest entry when full, so nothing has to be retired. Not a deque: a deque puts a block
    // malloc/free on the drain thread every few probes, and glibc hands large blocks back with munmap, which takes
    // mmap_lock for write and stalls every other thread in the process.
    std::array<Anchor, kProbeHistoryCapacity> probe_history_{};
    uint64_t probes_end_ = 0;

    std::optional<Chord> chord_;
    std::chrono::nanoseconds last_sync_error_{};
};

// Reads the profiler core's cycle counter and feeds the probes into the RealtimeProfilerClockMapping it owns. Nothing
// runs on device for this: the NOC serves the counter directly, so a read cannot be delayed by the profiler core's
// push loop.
//
// warm_up() runs before the receiver thread starts and every later call belongs to that thread, so nothing here is
// shared and none of it needs a publication protocol.
class RealtimeProfilerClockSync {
public:
    using Anchor = RealtimeProfilerClockMapping::Anchor;
    using RecordMapping = RealtimeProfilerClockMapping::RecordMapping;

    // Floor on how often each device's clock is read, under the probe every non-empty read already takes. Probe
    // spacing is the width of the pair a record is placed between, and a rate step inside that pair misplaces it by
    // step * width / 4. AICLK only moves on the ARC firmware's 1ms DVFS timer (dvfs.c:DVFSChange in
    // tt-zephyr-platforms), so anything well under a millisecond resolves a clock that cannot have changed within it:
    // p90 sync error is flat from 100us out to 500us and only breaks upward at the tick. Wormhole pins the upper end,
    // failing the didt p99 limit at 1ms where Blackhole is still at 1.8us. Overridable via
    // TT_RT_PROFILER_SYNC_INTERVAL_US.
    static std::chrono::nanoseconds sync_interval();

    // Extra reads taken only while the bracket is still wider than reads have recently been coming back at. Fires
    // rarely -- reads per resync reads 1.00 to two decimals -- but the ones it does take are the wide reads, and the
    // widest bracket is what the error bound is made of: removing it moved stress sync error p99 from 0.63us to
    // 5.05us and max from 1.28us to 25.06us.
    static constexpr int kResyncProbes = 4;

    // What the sync path costs its caller. `busy` is wall time inside resync(), which is dominated by the blocking
    // clock read, so on the receiver thread it is time not spent draining.
    struct Cost {
        uint64_t resyncs = 0;
        // Reads taken to satisfy those resyncs. Reported to enough precision to see the rare second read, since that
        // is what kResyncProbes exists for.
        uint64_t clock_reads = 0;
        std::chrono::nanoseconds busy{};

        Cost& operator+=(const Cost& o) {
            resyncs += o.resyncs;
            clock_reads += o.clock_reads;
            busy += o.busy;
            return *this;
        }
        [[nodiscard]] Cost since(const Cost& earlier) const {
            return Cost{resyncs - earlier.resyncs, clock_reads - earlier.clock_reads, busy - earlier.busy};
        }
    };

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

    // False when no UC window could be mapped onto the clock register. There is no slower path to fall back to: the
    // generic register read holds a chip-wide mutex and rewrites the window's configuration over PCIe on every call,
    // which lands inside the bracket that is the whole error bound, so a device without a window is not profiled.
    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    // Takes a few spaced probes so the first record already has a pair to be placed between, and a third to read the
    // clock's departure from. Runs before the receiver thread starts.
    void warm_up();

    // Takes one probe and retains it, returning what the clock read blocked the calling thread for -- on the receiver
    // thread that is time not spent draining. Cannot fail: the read is a load through an already-mapped window, and a
    // device without one is refused at construction, so the probe history only grows and a caller that has just
    // resynced can take a usable pair for granted.
    std::chrono::nanoseconds resync();

    // Whether this device is due a probe on the interval floor, having drained nothing to trigger one.
    [[nodiscard]] bool due_for_probe(std::chrono::steady_clock::time_point now) const {
        return now - last_probe_at_ >= sync_interval();
    }

    [[nodiscard]] std::optional<RecordMapping> map_record(uint64_t start_ticks, uint64_t end_ticks) {
        return mapping_.map_record(start_ticks, end_ticks);
    }

    // sync_error of the last record mapped, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_published_sync_error() const { return mapping_.last_sync_error(); }

    [[nodiscard]] Cost cost() const { return cost_; }

private:
    void configure_clock_read_path();
    // Placed at the midpoint of its read's bracket, which is where the counter could have been read.
    Anchor probe();
    // Ranked, not thresholded: under record load the whole bracket distribution shifts.
    Anchor best_of(int probes);

    ContextId context_id_;
    uint32_t chip_id_ = 0;
    // Resolved once so the resolve does not sit inside the bracket.
    CoreCoord profiler_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;
    // A UC window makes a probe a plain load. Required, not preferred: see has_direct_clock_read.
    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;
    // Recent tightest-read bracket, as an EMA. best_of stops early against this rather than an absolute target: the
    // whole bracket distribution shifts with record load, so a fixed target would never be met under load or never
    // filter when quiet.
    std::chrono::nanoseconds typical_bracket_{};

    // The high word only advances when the low word wraps, every 3.2s at ~1.35GHz.
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;
    Cost cost_;

    RealtimeProfilerClockMapping mapping_;
};

}  // namespace tt::tt_metal
