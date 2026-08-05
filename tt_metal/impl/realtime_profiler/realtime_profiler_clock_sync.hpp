// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
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

// host_ns = (device_ticks - device_cycle_offset) / frequency. frequency is fit once at bring-up and held fixed;
// device_cycle_offset is re-anchored whenever a probe finds the mapping has moved.
class RealtimeProfilerClockModel {
public:
    struct FitResidual {
        double rms_ns = 0.0;  // in device time
        double max_ns = 0.0;
        size_t num_probes_fitted = 0;   // regressed after discarding wide brackets
        size_t num_probes_offered = 0;  // handed to fit()
    };

    void seed_frequency(double frequency);

    // Empty when there were fewer than two probes to regress; the seeded frequency then stands, though a lone probe
    // is still anchored on.
    std::optional<FitResidual> fit(
        std::span<const ClockProbe> probes, std::chrono::steady_clock::time_point host_start);

    // Re-anchors only if the probe would place the mapping better than its current error.
    bool try_reanchor(const ClockProbe& probe);

    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping() const;

    // Placement error of the standing anchor plus the drift last measured against it. The single definition of sync
    // error; mapping() publishes exactly this.
    [[nodiscard]] std::chrono::nanoseconds sync_error() const;

    [[nodiscard]] double frequency() const { return frequency_; }

    // Zero before the first anchor.
    [[nodiscard]] std::chrono::nanoseconds anchor_bracket() const { return bracket_; }

    [[nodiscard]] std::chrono::nanoseconds last_drift() const { return last_drift_; }

    // What the standing mapping is out by at `probe`, without moving the anchor.
    [[nodiscard]] std::chrono::nanoseconds drift_at(const ClockProbe& probe) const;

    [[nodiscard]] bool is_anchored() const { return last_reanchor_at_.has_value(); }

    // Adopts a measured chord -- its slope is the local AICLK and its endpoint is a real probe, so this replaces both
    // halves of the mapping at once. Slope and offset must move together: offset is ticks - frequency * host_ns over a
    // host_ns of ~1e14, so a new slope against a stale offset misses by seconds. Ignored unless the slope lands in the
    // band around the commanded clock, so one bad pair of reads cannot mismap every record after it.
    void adopt_chord(double rate, std::chrono::steady_clock::time_point host_time, uint64_t device_ticks);

private:
    void set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks);

    double frequency_ = 0.0;
    double seed_frequency_ = 0.0;
    int64_t device_cycle_offset_ = 0;
    std::chrono::nanoseconds bracket_{};
    std::chrono::nanoseconds last_drift_{};
    std::optional<std::chrono::steady_clock::time_point> last_reanchor_at_;
};

// Reads the profiler core's cycle counter and drives a RealtimeProfilerClockModel from the probes. Nothing runs on
// device for this: the NOC serves the counter directly, so a read cannot be delayed by the profiler core's push loop.
//
// bring_up() runs before the receiver thread starts and every later call belongs to that thread, so nothing here is
// shared and the mapping needs no publication protocol.
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

    // What the sync path costs its caller. `busy` is wall time inside resync(), which is dominated by blocking
    // clock reads, so on the receiver thread it is time not spent draining.
    struct Cost {
        uint64_t resyncs = 0;
        uint64_t clock_reads = 0;
        std::chrono::nanoseconds busy{};
        // The two terms sync_error() sums, tracked apart: which one dominates decides whether a tail is the clock
        // moving or the read path being slow.
        std::chrono::nanoseconds bracket_sum{};
        std::chrono::nanoseconds bracket_max{};
        std::chrono::nanoseconds drift_sum{};
        std::chrono::nanoseconds drift_max{};
    };

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

    // The commanded AICLK stands if every fit attempt fails, so the mapping is usable either way.
    void bring_up();

    // One probe, offered to the model. False only when the device did not answer.
    bool resync();

    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping() const { return model_.mapping(); }

    [[nodiscard]] double frequency() const { return model_.frequency(); }

    // Two probes bracket an interval, and the secant between them maps any device timestamp inside it without
    // assuming anything about what the clock did in between.
    struct Anchor {
        std::chrono::steady_clock::time_point host;
        uint64_t ticks = 0;
        std::chrono::nanoseconds bracket{};
    };

    // Probes are retained, newest last, for exactly as long as a record might still need one as its near side: a
    // record is drained well after it ran, so the probe standing when it arrives is usually *after* it, and keeping
    // only that one would force every late record to be extrapolated backwards. There is no fixed depth -- the owner
    // retires them against the oldest record it still holds, so the history is as deep as the outstanding work makes
    // it and no deeper.

    // The tightest pair of probes around [start_ticks, end_ticks]. Nullopt only while no probe has yet read past the
    // record, which is the one thing worth waiting for. If the history no longer reaches back before the record, the
    // oldest probe still held is used as the near side rather than refusing: a record that can never be bracketed must
    // still go out, or staging never drains and the backlog feeds itself.
    [[nodiscard]] std::optional<std::pair<Anchor, Anchor>> probes_bracketing(
        uint64_t start_ticks, uint64_t end_ticks) const;

    [[nodiscard]] std::optional<Anchor> last_probe() const {
        return probes_.empty() ? std::nullopt : std::optional<Anchor>(probes_.back());
    }

    // Drops probes no record can need any more: everything before the newest one at or preceding `ticks`, which is the
    // oldest timestamp its owner still has staged. Called after publishing, so the retained span tracks the backlog.
    void retire_probes_before(uint64_t ticks);

    // What a closed interval publishes its records with.
    struct ChordMapping {
        experimental::ProgramRealtimeClockSync mapping;
        double frequency = 0.0;
        // Uncertainty in `frequency`, as a fraction: the two probes' brackets over the span they were measured across.
        double rate_noise = 0.0;
    };

    // Chooses what a closed interval publishes with, given its two probes and the previous interval's slope, or
    // nullopt when the pair cannot be taken as a chord at all. Pure by construction -- no device, no clock, no state --
    // so the judgements it makes (when a chord is too short to take a slope from, how much of a rate change is real,
    // what error the records carry) are testable without silicon, which is the only way they get to be trustworthy.
    [[nodiscard]] static std::optional<ChordMapping> plan_chord_mapping(
        const Anchor& open,
        const Anchor& closing,
        double previous_rate,
        double previous_rate_noise,
        double sanity_rate);

    // Error a device timestamp interpolated between `open` and `close` carries: it lands on the secant through two
    // measured points, so it can only be out by however far those points themselves are placed.
    [[nodiscard]] static std::chrono::nanoseconds interpolation_error(const Anchor& open, const Anchor& close) {
        return std::max(open.bracket, close.bracket) / 2;
    }

    [[nodiscard]] Cost cost() const { return cost_; }

    void adopt_chord(double rate, std::chrono::steady_clock::time_point host_time, uint64_t device_ticks) {
        model_.adopt_chord(rate, host_time, device_ticks);
    }

private:
    // False when there is no usable cache entry, or the anchor probe failed and a full fit is needed.
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
    // Recent tightest-read bracket, as an EMA. best_of stops early against this rather than an absolute target
    // because the whole bracket distribution shifts with record load, so a fixed target would either never be met
    // under load or never filter when quiet.
    std::chrono::nanoseconds typical_bracket_{};

    RealtimeProfilerClockModel model_;
};

}  // namespace tt::tt_metal
