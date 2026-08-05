// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <utility>

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

    // The one tunable in the sync path: how often a probe is taken. Every probe closes the interval its records ran
    // in, so this single number sets all three things that trade against each other -- delivery latency (a record
    // waits at most this long to be published), interpolation error (a DVFS step inside an interval costs the records
    // in it up to rate_change * interval / 4), and cost (one blocking clock read per device per interval, ~0.7us of
    // PCIe stall). Raising it is cheaper and slower in equal measure.
    static constexpr auto kSyncInterval = std::chrono::microseconds(100);

    // Ranking a few reads and keeping the tightest keeps the anchor's settling error near the read floor rather
    // than whatever the link was doing at the time: on a 32-chip mesh a lone read brackets 11-36us against
    // 0.7-0.9us here.
    static constexpr int kResyncProbes = 4;

    // What the sync path costs its caller. `busy` is wall time inside resync(), which is dominated by the blocking
    // clock read, so on the receiver thread it is time not spent draining.
    struct Cost {
        uint64_t resyncs = 0;
        uint64_t clock_reads = 0;
        std::chrono::nanoseconds busy{};
    };

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

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

    // The tightest pair of probes around [start_ticks, end_ticks], or nullopt while no probe has yet read past the
    // record.
    [[nodiscard]] std::optional<std::pair<Anchor, Anchor>> probes_bracketing(
        uint64_t start_ticks, uint64_t end_ticks) const;

    // Drops probes older than `ticks`, the oldest timestamp its owner still holds. This is what bounds the history:
    // a probe is kept only while some record might still need it as a near side.
    void retire_probes_before(uint64_t ticks);

    // What a closed interval publishes its records with.
    struct ChordMapping {
        experimental::ProgramRealtimeClockSync mapping;
        double frequency = 0.0;
        // Uncertainty in `frequency`, as a fraction: the two probes' brackets over the span they were measured across.
        double rate_noise = 0.0;
    };

    // Chooses what a closed interval publishes with, given its two probes and the previous interval's slope, or
    // nullopt when the pair cannot be taken as a chord at all. Pure: no device, no clock, no state.
    [[nodiscard]] static std::optional<ChordMapping> plan_chord_mapping(
        const Anchor& open,
        const Anchor& closing,
        double previous_rate,
        double previous_rate_noise,
        double sanity_rate);

    // The endpoint term of an interpolated timestamp's error: it lands on the secant through two measured points, so
    // it inherits how well those points are placed. A clock that moved within the interval adds to this; see the
    // curvature term in plan_chord_mapping.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    [[nodiscard]] Cost cost() const { return cost_; }

    void adopt_rate(double rate) { model_.adopt_rate(rate); }

private:
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
    // A UC window makes a probe a load rather than a UMD call, whose TLB reconfiguration would sit inside the
    // bracket and widen it by ~450ns.
    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;
    // The high word only advances when the low word wraps, every 3.2s at ~1.35GHz.
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;
    Cost cost_;
    std::deque<Anchor> probes_;
    double wrap_period_frequency_ = 0.0;
    std::chrono::nanoseconds wrap_period_{};
    // Recent tightest-read bracket, as an EMA. best_of stops early against this rather than an absolute target: the
    // whole bracket distribution shifts with record load, so a fixed target would never be met under load or never
    // filter when quiet.
    std::chrono::nanoseconds typical_bracket_{};

    RealtimeProfilerClockModel model_;
};

}  // namespace tt::tt_metal
