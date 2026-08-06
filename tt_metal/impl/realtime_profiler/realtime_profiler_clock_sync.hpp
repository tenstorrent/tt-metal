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

// The device clock's rate in ticks per host nanosecond: regressed over the bring-up probes, then re-measured from
// each chord the receiver closes. Records are mapped from their own pair of probes, not from here -- this rate is
// what a chord's slope is sanity-checked against, and what stands in before the first chord exists.
class RealtimeProfilerClockModel {
public:
    struct FitResidual {
        double rms_ns = 0.0;  // in device time
        double max_ns = 0.0;
        size_t num_probes_fitted = 0;   // regressed after discarding wide brackets
        size_t num_probes_offered = 0;  // handed to fit()
    };

    void seed_frequency(double frequency);

    // Empty when there were fewer than two probes to regress; the seeded frequency then stands.
    std::optional<FitResidual> fit(
        std::span<const ClockProbe> probes, std::chrono::steady_clock::time_point host_start);

    [[nodiscard]] double frequency() const { return frequency_; }

    // Adopts a rate measured elsewhere -- the secant across a closed interval, which is the local AICLK. Ignored
    // unless it lands in the band around the commanded clock, so one bad pair of reads cannot mismap the records that
    // follow.
    void adopt_rate(double rate);

private:
    double frequency_ = 0.0;
    double seed_frequency_ = 0.0;
};

// Reads the profiler core's cycle counter and keeps the probes a record needs to be interpolated between. Nothing
// runs on device for this: the NOC serves the counter directly, so a read cannot be delayed by the profiler core's
// push loop.
//
// bring_up() runs before the receiver thread starts and every later call belongs to that thread, so nothing here is
// shared and none of it needs a publication protocol.
class RealtimeProfilerClockSync {
public:
    // How far a measured rate may sit from the commanded AICLK and still be believed. Guards every place a rate is
    // adopted or a chord is sanity-checked, so one bad pair of reads cannot mismap the records that follow.
    static constexpr double kRateClampFraction = 0.10;

    // How often a probe is taken, per device. Every probe closes the interval its records ran in, so this one number
    // sets delivery latency (a record waits at most this long to be published), the interpolation error a DVFS step
    // inside an interval leaves (up to rate_change * interval / 4), and cost -- one blocking PCIe read per device per
    // interval, measured at 891ns, so 32 devices at 100us is a third of a core.
    //
    // AICLK only moves when the ARC firmware's DVFS loop runs, which is a 1ms timer (dvfs.c:DVFSChange in
    // tt-zephyr-platforms), so probes spaced far below that re-measure a clock that provably cannot have changed --
    // measured on Blackhole under didt, p90 sync error is flat at 385-390ns from 100us all the way out to 500us and
    // only breaks upward at the tick. 500us is where both parts sit comfortably: it is Wormhole that pins it, failing
    // the 15us p99 limit at 1ms (17.6us) where Blackhole is still at 1.8us. Overridable via
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
    };

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

    // False when no UC window could be mapped onto the clock register. There is no slower path to fall back to: the
    // generic register read holds a chip-wide mutex and rewrites the window's configuration over PCIe on every call,
    // which lands inside the bracket that is the whole error bound, so a device without a window is not profiled.
    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    // The commanded AICLK stands if every fit attempt fails, so the mapping is usable either way.
    void bring_up();

    // Takes one probe and retains it. False only when the device did not answer.
    bool resync();

    [[nodiscard]] double frequency() const { return model_.frequency(); }

    // A probe, placed at the midpoint of the bracket its read fell in. Two of them map any device timestamp between
    // them via their secant, whatever the clock did in between.
    struct Anchor {
        std::chrono::steady_clock::time_point host;
        uint64_t ticks = 0;
        std::chrono::nanoseconds bracket{};
    };

    // The device timestamp of the oldest probe that has read past `ticks`, or nullopt while none has. A record is only
    // publishable once its end is covered: that is what says the clock is known on both sides of it. Receiver thread.
    [[nodiscard]] std::optional<uint64_t> coverage_past(uint64_t ticks) const;

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
        // This interval's own slope. Published to nobody: it is the local rate, so it is what the next interval's
        // curvature term has to be compared against, and a baseline rate would have smoothed the step away.
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

    // Chooses what a closed interval publishes with, given its two probes and the previous interval's slope, or
    // nullopt when the pair cannot be taken as a chord at all. Pure: no device, no clock, no state.
    [[nodiscard]] static std::optional<ChordMapping> plan_chord_mapping(
        const Anchor& open,
        const Anchor& closing,
        const std::optional<BaselineRate>& baseline,
        double previous_rate,
        double previous_rate_noise,
        double sanity_rate);

    // What to publish `ticks` with: the tightest pair of probes around it that can be taken as a chord, else the single
    // probe past it sloped at the fitted rate. Nullopt only when no probe has read past `ticks` at all, so anything
    // coverage_past admits, this places.
    //
    // Total on purpose, and it degrades rather than refusing. This is asked about the oldest staged record repeatedly
    // and its inputs do not change between asks, so a refusal is not retried -- it stands until the ring laps the probe
    // it refused, seconds later, with every record behind it held up because records publish in order. Receiver thread.
    [[nodiscard]] std::optional<ChordMapping> place(uint64_t ticks);

    // sync_error of the last interval this closed, which is the bound currently standing for this device.
    [[nodiscard]] std::chrono::nanoseconds last_published_sync_error() const { return last_published_sync_error_; }

    // The endpoint term of an interpolated timestamp's error: it lands on the secant through two measured points, so
    // it inherits how well those points are placed. A clock that moved within the interval adds to this; see the
    // curvature term in plan_chord_mapping.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    [[nodiscard]] Cost cost() const { return cost_; }

private:
    // How many successively wider near anchors place() tries before it gives up on measuring a slope at all. Both of
    // plan_chord_mapping's refusals scale as 1/span, so widening is what clears them; past a few steps it is the far
    // anchor that does not fit, and no near anchor repairs that.
    static constexpr uint64_t kPlacementWidenSteps = 8;

    // Index of the oldest retained probe whose counter read reached `ticks`, or probes_end_ when none has. Probes are
    // appended in tick order, so this bisects: the retained span grows with the backlog, and scanning it per record is
    // what turns a backlog into a stall.
    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t ticks) const;

    // False when there is no usable cache entry, or the probe failed and a full fit is needed.
    bool try_cached_calibration();
    // False when the fit is not worth keeping and another attempt is likely to beat it.
    bool calibrate();
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

    // The last two intervals' slopes. Only the difference between consecutive ones says the clock moved *within* one,
    // so `previous_rate_` is what the curvature term is measured against.
    //
    // Which interval is which is keyed on the closing anchor rather than on call order, because an interval is closed
    // once per drain pass until the next probe arrives -- passes run every few hundred microseconds against a 500us
    // interval -- and rolling these forward per call would leave a chord compared against itself, reading zero
    // curvature for every pass after the first.
    uint64_t current_chord_close_ticks_ = 0;
    double current_rate_ = 0.0;
    double current_rate_noise_ = 0.0;
    double previous_rate_ = 0.0;
    double previous_rate_noise_ = 0.0;
    std::chrono::nanoseconds last_published_sync_error_{};

    double wrap_period_frequency_ = 0.0;
    std::chrono::nanoseconds wrap_period_{};

    RealtimeProfilerClockModel model_;
};

}  // namespace tt::tt_metal
