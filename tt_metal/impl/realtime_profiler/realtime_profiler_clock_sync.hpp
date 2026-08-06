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
#include <span>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

struct ClockProbe {
    std::chrono::steady_clock::time_point host_time;  // taken immediately before the counter read
    std::chrono::nanoseconds bracket{};
    uint64_t device_ticks = 0;
};

// Reads the profiler core's cycle counter and keeps the probes a record needs to be interpolated between. Nothing
// runs on device for this: the NOC serves the counter directly, so a read cannot be delayed by the profiler core's
// push loop.
//
// warm_up() runs before the receiver thread starts and every later call belongs to that thread, so nothing here is
// shared and none of it needs a publication protocol.
class RealtimeProfilerClockSync {
public:
    // The baseline a chord's own slope has to be measured across, as twice the floor: a pair closer together than half
    // of this is refused and the near anchor is taken from further back. It no longer sets a probe cadence -- probes
    // are taken by whoever reads records, right after reading them -- so it costs no PCIe traffic and buys only how
    // well a local slope is resolved.
    //
    // AICLK only moves when the ARC firmware's DVFS loop runs, which is a 1ms timer (dvfs.c:DVFSChange in
    // tt-zephyr-platforms), so a chord spanning far below that resolves a clock that provably cannot have changed
    // within it -- measured on Blackhole under didt, p90 sync error is flat at 385-390ns from 100us all the way out to
    // 500us and only breaks upward at the tick. 500us is where both parts sit comfortably: it is Wormhole that pins it,
    // failing the 15us p99 limit at 1ms (17.6us) where Blackhole is still at 1.8us. Overridable via
    // TT_RT_PROFILER_SYNC_INTERVAL_US.
    static std::chrono::nanoseconds sync_interval();

    // Ranking a few reads and keeping the tightest holds the anchor's settling error near the read floor rather than
    // wherever the link happened to be. It fires rarely -- reads per resync reads 1.00 to two decimals -- but the reads
    // it does take are the wide ones, and they are what the error bound is made of: removing this moved stress sync
    // error p99 from 0.63us to 5.05us, max from 1.28us to 25.06us, and the fitted frequency from 0.25ppm to 3.5ppm.
    static constexpr int kResyncProbes = 4;

    // How wide a baseline the rate a record is published with is measured across. A chord's slope is uncertain by
    // (both brackets)/span, so the ~100us chord a record sits in carries a few thousand ppm -- measured 600ppm rms and
    // 3800ppm p1-p99 on an 8-chip mesh -- and that lands on every duration a consumer divides out. The same probes
    // spanning 4ms give ~500ppm, which is still an order of magnitude tighter than the ~5200ppm AICLK steps the rate
    // has to keep tracking, so widening it further would start smoothing over real DVFS instead of noise.
    static constexpr auto kRateBaseline = std::chrono::milliseconds(4);

    // What the sync path costs its caller. `busy` is wall time inside resync(), which is dominated by the blocking
    // clock read, so on the receiver thread it is time not spent draining.
    struct Cost {
        uint64_t resyncs = 0;
        // Reads taken to satisfy those resyncs. Reported to enough precision to see the rare second read, since that is
        // what kResyncProbes exists for.
        uint64_t clock_reads = 0;
        std::chrono::nanoseconds busy{};
        // Chords resolved, and how many of them had a probe inside to measure the clock's departure against. The bow
        // term is silent on a settled clock, which is correct and indistinguishable from a term that never had any
        // evidence to work with -- the failure the inferred curvature it replaced went unnoticed for. Reported so the
        // difference is visible without a throttling workload.
        uint64_t chords_placed = 0;
        uint64_t chords_with_interior_probe = 0;
    };

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

    // False when no UC window could be mapped onto the clock register. There is no slower path to fall back to: the
    // generic register read holds a chip-wide mutex and rewrites the window's configuration over PCIe on every call,
    // which lands inside the bracket that is the whole error bound, so a device without a window is not profiled.
    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    // Takes a few spaced probes so that baseline_rate() is already measurable when the first record arrives. Runs
    // before the receiver thread starts; a device that cannot be probed keeps the commanded AICLK, which is what the
    // fallback in place() is for.
    void warm_up();

    // Takes one probe and retains it. False only when the device did not answer.
    bool resync();

    // The last rate measured across kRateBaseline, or the commanded AICLK before any has been. Only a timestamp the
    // probe history does not surround is published at this rate; anything it does surround rides its own chord.
    [[nodiscard]] double frequency() const { return fallback_rate_; }

    // A probe, placed at the midpoint of the bracket its read fell in. Two of them map any device timestamp between
    // them via their secant, whatever the clock did in between.
    struct Anchor {
        std::chrono::steady_clock::time_point host;
        uint64_t ticks = 0;
        std::chrono::nanoseconds bracket{};
    };

    // A rate measured across the whole retained history rather than across one chord.
    struct BaselineRate {
        double rate = 0.0;
        // Uncertainty in `rate`, as a fraction: the two probes' brackets over the span they were measured across.
        double noise = 0.0;
    };

    // The rate across the retained probe history, or nullopt while it is too narrow to beat a single chord. Receiver
    // thread, like the rest of the probe history.
    [[nodiscard]] std::optional<BaselineRate> baseline_rate() const;

    // What a closed interval publishes its records with. `mapping` is per record, so it is left for place_on_chord;
    // everything else is shared by every record the chord covers.
    struct ChordMapping {
        experimental::ProgramRealtimeClockSync mapping;
        // Measured across kRateBaseline, not across this chord.
        double frequency = 0.0;
        // This chord's own slope, published to nobody: placement rides it, so that a rate measured across a wider
        // baseline cannot move where a record lands. See place_on_chord.
        double chord_rate = 0.0;
        // Uncertainty in `chord_rate`, as a fraction: the two brackets over the span they were measured across.
        double chord_rate_noise = 0.0;
        // Enough to place a timestamp on the chord: its near anchor, and the reciprocal slope, inverted once here
        // because the alternative is a division per record on the drain thread.
        uint64_t open_ticks = 0;
        double open_host_ns = 0.0;
        double inv_chord_rate = 0.0;
        // Not needed to place anything; it is what says whether a timestamp is being interpolated between the two
        // measured points or extrapolated past them. See place_on_chord.
        uint64_t close_ticks = 0;
        // The largest start timestamp this mapping may be reused for. A measured chord stops at its far anchor, because
        // past that a tighter pair exists or will. An extrapolated one has no such limit: place_on_chord already
        // charges each timestamp for its own distance from the anchor, so one of them stamps a whole backlog correctly.
        uint64_t batch_through_ticks = 0;
    };

    // The offset that restates a record's own interpolated placement in terms of the published `frequency`. Anchoring
    // per record rather than once per chord is what keeps a baseline-wide rate from costing anything: the record's
    // start lands exactly where the chord puts it, and only its duration -- microseconds, not the chord's span -- is
    // carried at a rate that may differ from the local one.
    [[nodiscard]] static int64_t device_cycle_offset_for(const ChordMapping& chord, uint64_t start_ticks) {
        const double host_ns =
            chord.open_host_ns +
            (static_cast<double>(start_ticks) - static_cast<double>(chord.open_ticks)) * chord.inv_chord_rate;
        return std::llround(static_cast<double>(start_ticks) - chord.frequency * host_ns);
    }

    // Where `start_ticks` lands on `chord`, and what that placement is worth. Between the two anchors the secant cannot
    // be further out than the worse of them whatever its slope, so the chord's own bound stands unchanged. Past them
    // the slope is extrapolated and its uncertainty becomes distance * noise, unbounded: a timestamp placed from a
    // chord a second away is wrong by milliseconds, and the chord's bound would report sub-microsecond.
    [[nodiscard]] static experimental::ProgramRealtimeClockSync place_on_chord(
        const ChordMapping& chord, uint64_t start_ticks) {
        uint64_t outside = 0;
        if (start_ticks < chord.open_ticks) {
            outside = chord.open_ticks - start_ticks;
        } else if (start_ticks > chord.close_ticks) {
            outside = start_ticks - chord.close_ticks;
        }
        const auto extrapolation = std::chrono::nanoseconds(
            static_cast<int64_t>(static_cast<double>(outside) * chord.inv_chord_rate * chord.chord_rate_noise));
        return experimental::ProgramRealtimeClockSync{
            .device_cycle_offset = device_cycle_offset_for(chord, start_ticks),
            .sync_error = chord.mapping.sync_error + extrapolation,
        };
    }

    // Chooses what a chord publishes with, or nullopt when the pair cannot be taken as a chord at all. `measured_bow`
    // is how far the clock was seen to depart from this chord, from measured_bow(); it lands in sync_error on top of
    // the endpoint term. Pure: no device, no clock, no state.
    [[nodiscard]] static std::optional<ChordMapping> plan_chord_mapping(
        const Anchor& open,
        const Anchor& closing,
        const std::optional<BaselineRate>& baseline,
        std::chrono::nanoseconds measured_bow);

    // What to publish `ticks` with: the tightest pair of retained probes around it, or a single anchor extrapolated
    // from when the history does not surround it. Total except with no probe retained at all, so a caller that has just
    // probed always gets an answer and nothing is ever held back for a later pass.
    //
    // Asked once per record, on the pass that read it. Never conditional on the device answering -- that is what let a
    // refusal here stall a device's whole data path, because it was asked repeatedly about the same record with inputs
    // that did not change between asks. Receiver thread.
    [[nodiscard]] std::optional<ChordMapping> place(uint64_t ticks);

    // sync_error of the last interval this closed, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_published_sync_error() const { return last_published_sync_error_; }

    // The endpoint term of an interpolated timestamp's error: it lands on the secant through two measured points, so
    // it inherits how well those points are placed. A clock that moved within the interval adds to this; see
    // measured_bow.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    // How far `interior` lies off the chord through `open` and `close`, less its own read noise. A probe inside a chord
    // was not fitted to it, so this is the clock's departure at that point -- measured, where the alternative is to
    // infer it from how much two chords' slopes differ and hope the difference was the clock rather than the reads.
    // Pure.
    [[nodiscard]] static std::chrono::nanoseconds departure_from_chord(
        const Anchor& open, const Anchor& close, const Anchor& interior);

    [[nodiscard]] Cost cost() const { return cost_; }

private:
    // What the retained probes say the clock did between two of them that the chord through them does not capture. Zero
    // when no probe lies inside the chord, which is the absence of evidence, not a claim of linearity.
    [[nodiscard]] std::chrono::nanoseconds measured_bow(uint64_t open_index, uint64_t close_index) const;

    // A mapping pinned to one probe and sloped at the best rate available, for a timestamp the probe history does not
    // surround. Nullopt only if no rate is known at all.
    [[nodiscard]] std::optional<ChordMapping> extrapolate_from(
        const Anchor& anchor, const std::optional<BaselineRate>& baseline);

    // Index of the oldest retained probe whose counter read reached `ticks`, or probes_end_ when none has. Probes are
    // appended in tick order, so this bisects: the retained span grows with the backlog, and scanning it per record is
    // what turns a backlog into a stall.
    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t ticks) const;

    void configure_clock_read_path();
    std::optional<ClockProbe> probe();
    // Ranked, not thresholded: under record load the whole bracket distribution shifts.
    std::optional<ClockProbe> best_of(int probes);

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

    // Preallocated, and overwrites its oldest entry when full, so nothing has to be retired: the live range is always
    // the newest kProbeHistoryCapacity probes and its start is derived. Not a deque, because a deque puts a block
    // malloc/free on the drain thread every few probes and this thread must never touch the allocator -- glibc hands
    // large blocks back with munmap, which takes mmap_lock for write and stalls every other thread in the process, so
    // an allocation here couples the drain loop to whatever every consumer thread is doing.
    //
    // Sized far past what is needed: a probe within one interval closes a record, and the published rate spans
    // kRateBaseline, so at any interval this is configurable to the entries either could still want are present many
    // times over.
    static constexpr size_t kProbeHistoryCapacity = 4096;
    std::array<Anchor, kProbeHistoryCapacity> probe_history_{};
    uint64_t probes_end_ = 0;

    [[nodiscard]] uint64_t oldest_probe() const {
        return probes_end_ > kProbeHistoryCapacity ? probes_end_ - kProbeHistoryCapacity : 0;
    }
    [[nodiscard]] const Anchor& probe_at(uint64_t index) const { return probe_history_[index % kProbeHistoryCapacity]; }

    std::chrono::nanoseconds last_published_sync_error_{};

    double fallback_rate_ = 0.0;
};

}  // namespace tt::tt_metal
