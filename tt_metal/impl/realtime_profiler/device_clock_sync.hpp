// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

// Probe spacing floor. Certification needs two adjacent probe gaps plus read noise to fit inside
// kDvfsMinTransitionSpacing, so the floor sits well under half of it to leave jitter margin.
inline constexpr auto kDeviceClockSyncInterval = std::chrono::microseconds(250);

// Minimum spacing between starts of device clock-rate transitions. AICLK is stepped by the ARC
// firmware's 1 ms DVFS timer, one monotone PLL glide per tick; a board-power message can pull a
// tick up to ~10 us early, and the 50 us margin absorbs several such pulls stacking on one tick.
// Verified against Blackhole firmware (tt-system-firmware dvfs.c/pll driver); assumed for
// Wormhole. Host-forced clock operations (forced AICLK/VDD, AICLK sweep, clock-scheme switch)
// bypass the timer and void this constant.
inline constexpr auto kDvfsMinTransitionSpacing = std::chrono::microseconds(950);

// Maps device cycle-counter timestamps onto the host steady_clock timeline by interpolating
// between retained clock probes, publishing per record an affine mapping plus an error that
// upper-bounds the host-time placement error of both record endpoints.
//
// The error is a sound upper bound provided that: each probe's true read time lies within
// ±Anchor::error of its host_timestamp (bracketed read); the device counter is monotone and its
// rate piecewise-constant up to monotone transitions (DVFS glides) whose starts are at least
// kDvfsMinTransitionSpacing apart; no host-forced clock operation is active; and, if a
// FrequencyPrior is supplied, the rate never leaves it.
//
// A chord (the interval between adjacent probes) is *certified* once the two-chord windows on
// both sides of it measure shorter than kDvfsMinTransitionSpacing: at most one transition can
// then touch the chord, its neighbors are transition-free, their secant rates bracket every rate
// inside the chord, and the worst-case interpolation error follows from that bracket — a hard
// bound under the assumptions above alone. Chords that cannot be certified (receiver stalls,
// history edges) and records whose start predates the retained history are bounded against the
// lifetime-observed rate envelope instead (the FrequencyPrior until anything has been observed;
// the chord's own span without either). That tier adds one assumption: an unobserved window
// holds no rate the clock has never visited. It understates only when a first-ever rate
// excursion lands inside an unprobed gap; the alternative — the platform's full idle-to-limit
// range — never understates but overstates such records' error by ~50x.
//
// A chord's bound tightens when its successor probe lands (finalized_device_timestamp()); mapping
// a record whose end lies past that watermark is sound but yields the fallback-quality error.
//
// Not thread-safe: owned and driven by the receiver thread.
class ClockSyncMapping {
public:
    // A (host_timestamp, device_timestamp) sample and the uncertainty of that host time.
    struct Anchor {
        std::chrono::steady_clock::time_point host_timestamp;
        uint64_t device_timestamp = 0;
        std::chrono::nanoseconds error{};
    };

    // Affine map: host_ns = (device_timestamp - device_cycle_offset) / frequency.
    struct RecordMapping {
        int64_t device_cycle_offset = 0;
        std::chrono::nanoseconds error{};
        double frequency = 0.0;
    };

    // Rate range (device cycles per host ns) the clock cannot leave, e.g. the platform's
    // idle/limit AICLK values. Trusted as a hard bound; see the class contract.
    struct FrequencyPrior {
        double min_frequency = 0.0;
        double max_frequency = 0.0;
    };

    static constexpr size_t kProbeHistoryCapacity =
        std::bit_ceil(static_cast<size_t>(std::chrono::seconds(2) / kDeviceClockSyncInterval));

    explicit ClockSyncMapping(std::optional<FrequencyPrior> frequency_prior = std::nullopt);

    // Requires probe host/device times strictly after the previous retained probe.
    void add_probe(const Anchor& probe);

    // Snapshot a start timestamp onto the host timeline while probes still cover it (e.g.
    // long-running programs). Returns false until the chord around it is finalized, so callers
    // must keep offering the timestamp until it takes.
    bool pin_start(uint64_t device_timestamp);

    // Nullopt only when no retained probe precedes either timestamp.
    [[nodiscard]] std::optional<RecordMapping> map_record(
        uint64_t start_device_timestamp, uint64_t end_device_timestamp);

    // Newest device timestamp whose surrounding chord bounds are final. Zero until two probes
    // exist.
    [[nodiscard]] uint64_t finalized_device_timestamp() const {
        return probes_end_ >= 2 ? probe_at(probes_end_ - 2).device_timestamp : 0;
    }

    // Diagnostics: chords finalized, and how many of those earned the certificate (tight bounds).
    [[nodiscard]] uint64_t num_finalized_chords() const {
        return num_finalized_chords_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t num_certified_chords() const {
        return num_certified_chords_.load(std::memory_order_relaxed);
    }

    // Records whose bound came from an uncertified chord or the history envelope. They are still
    // published, with envelope-tier bounds — this only counts them, so callers can assert the
    // fallback tier stays a tiny fraction of traffic.
    [[nodiscard]] uint64_t num_records_on_uncertified_chords() const {
        return num_records_on_uncertified_chords_.load(std::memory_order_relaxed);
    }

private:
    // Secant between adjacent probes, plus the allowance for a rate transition inside it.
    // Error quantities are precomputed integers: map_record runs per record on the receiver hot
    // path, so everything derivable per chord is derived here (at probe rate, not record rate).
    struct Chord {
        int64_t probe_error_ns = 0;  // max of the two probes' half-brackets
        // Worst-case in-chord misplacement from one transition: span * (sqrt(rho)-1)/(sqrt(rho)+1).
        int64_t nonlinearity_ns = 0;
        // Distance-to-nearest-probe refinement: below this many cycles, refine_slope * distance
        // beats nonlinearity_ns; zero when no rate bracket exists (refinement disabled).
        uint64_t refine_threshold_cycles = 0;
        double refine_slope = 0.0;  // ns per cycle: 1/rate_lo - 1/rate_hi
        // Certified bracket on every rate inside this chord; zero when uncertified.
        double rate_lo = 0.0;
        double rate_hi = 0.0;
        double frequency = 0.0;           // secant of the adjacent probes; interpolation slope
        double smoothed_frequency = 0.0;  // frequency-window secant as of finalize; what records publish
        // |1/smoothed_frequency - 1/frequency|, ns per cycle: consumer host_end reconstruction skew.
        double smoothing_skew_per_cycle = 0.0;
        uint64_t open_device_timestamp = 0;
        double open_host_ns = 0.0;
        uint64_t close_device_timestamp = 0;
    };

    // Sliding baseline for the published frequency: at >=1 s the secant noise (read error / span)
    // sits at ~1 ppm; the max keeps the anchor safely inside the 2 s probe retention.
    static constexpr auto kFrequencyWindowSlide = std::chrono::seconds(1);
    static constexpr auto kFrequencyWindowMax = std::chrono::milliseconds(1600);

    [[nodiscard]] std::optional<uint64_t> chord_index_around(uint64_t device_timestamp) const;

    [[nodiscard]] static double host_ns_on(const Chord& chord, uint64_t device_timestamp) {
        return chord.open_host_ns +
               (static_cast<double>(device_timestamp) - static_cast<double>(chord.open_device_timestamp)) /
                   chord.frequency;
    }

    // Placement error bound in ns for one timestamp inside (or, defensively, outside) this chord.
    // Integer math; rounds up, never down.
    [[nodiscard]] static int64_t error_ns_on(const Chord& chord, uint64_t device_timestamp);

    void set_fallback_step_bound(Chord& chord, double span_ns) const;

    // Upgrades chord close_index from its fallback bound once both neighbors exist: evaluates the
    // certificate and computes the neighbor-bracket bound. Called when probe close_index+1 lands.
    void finalize_chord(uint64_t close_index);

    // Re-anchors the frequency window once it exceeds kFrequencyWindowMax, rebuilding the running
    // intersection over the chords it keeps (rare; the intersection of a subset cannot be empty).
    void slide_frequency_window(uint64_t close_index);

    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t device_timestamp) const;

    [[nodiscard]] uint64_t oldest_probe() const {
        return probes_end_ > kProbeHistoryCapacity ? probes_end_ - kProbeHistoryCapacity : 0;
    }
    [[nodiscard]] const Anchor& probe_at(uint64_t index) const {
        return probe_history_[index & (kProbeHistoryCapacity - 1)];
    }
    [[nodiscard]] const Chord& chord_at(uint64_t close_index) const {
        return chords_[close_index & (kProbeHistoryCapacity - 1)];
    }

    std::array<Anchor, kProbeHistoryCapacity> probe_history_{};
    std::array<Chord, kProbeHistoryCapacity> chords_{};  // chord ending at this probe index
    uint64_t probes_end_ = 0;
    mutable uint64_t last_probe_index_ = 0;

    std::optional<FrequencyPrior> frequency_prior_;

    // Widest noise-widened secant range ever measured; input to the history-exceeded envelope.
    double observed_min_frequency_ = 0.0;
    double observed_max_frequency_ = 0.0;

    // Frequency window: the run of consecutive certified chords whose rate brackets all intersect.
    // Maintaining the *running intersection* (not pairwise checks) makes hidden-step creep
    // impossible: every rate in the window provably lies inside it, so the window may grow
    // without bound and only a detected transition (empty intersection) resets it.
    bool frequency_window_active_ = false;
    uint64_t frequency_window_anchor_ = 0;  // probe index the window secant is measured from
    double frequency_window_rate_lo_ = 0.0;
    double frequency_window_rate_hi_ = 0.0;

    // Relaxed atomics so diagnostics accessors are readable from any thread; everything else on
    // this class is receiver-thread-only.
    std::atomic<uint64_t> num_finalized_chords_{0};
    std::atomic<uint64_t> num_certified_chords_{0};
    std::atomic<uint64_t> num_records_on_uncertified_chords_{0};

    // chord_index_around(start) cache.
    std::optional<uint64_t> active_chord_index_;

    // From pin_start; cleared once map_record uses it.
    std::optional<Anchor> pinned_start_;
    uint64_t last_pin_device_timestamp_ = 0;
};

// Reads a tensix free-running cycle counter over PCIe (bracketed between steady_clock reads) and
// feeds the probes to a ClockSyncMapping it owns.
class DeviceClockSync {
public:
    using Anchor = ClockSyncMapping::Anchor;
    using RecordMapping = ClockSyncMapping::RecordMapping;

    static constexpr size_t kProbeHistoryCapacity = ClockSyncMapping::kProbeHistoryCapacity;

    static constexpr int kResyncProbes = 4;

    DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core);
    ~DeviceClockSync();

    [[nodiscard]] bool has_direct_clock_read() const { return mapped_clock_lo_ != nullptr; }

    void resync();

    [[nodiscard]] bool due_for_probe(std::chrono::steady_clock::time_point now) const {
        return now - last_probe_at_ >= kDeviceClockSyncInterval;
    }

    [[nodiscard]] std::chrono::steady_clock::time_point next_probe_due() const {
        return last_probe_at_ + kDeviceClockSyncInterval;
    }

    // Snapshot a start timestamp onto the host timeline while probes still cover it (e.g. long-running programs).
    // Returns false while the surrounding chord is not yet finalized; callers re-offer the timestamp.
    bool pin_start(uint64_t device_timestamp) { return mapping_.pin_start(device_timestamp); }

    // Nullopt only when no retained probe precedes either timestamp.
    [[nodiscard]] std::optional<RecordMapping> map_record(
        uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
        return mapping_.map_record(start_device_timestamp, end_device_timestamp);
    }

    // See ClockSyncMapping::finalized_device_timestamp().
    [[nodiscard]] uint64_t finalized_device_timestamp() const { return mapping_.finalized_device_timestamp(); }

    [[nodiscard]] uint64_t num_finalized_chords() const { return mapping_.num_finalized_chords(); }
    [[nodiscard]] uint64_t num_certified_chords() const { return mapping_.num_certified_chords(); }
    [[nodiscard]] uint64_t num_records_on_uncertified_chords() const {
        return mapping_.num_records_on_uncertified_chords();
    }

    // Largest gap between consecutive probes since the last call; reading clears it. Callable from
    // any thread.
    [[nodiscard]] std::chrono::nanoseconds take_peak_probe_gap() {
        return std::chrono::nanoseconds(peak_probe_gap_ns_.exchange(0, std::memory_order_relaxed));
    }

private:
    void configure_clock_read_path();

    Anchor probe();

    ContextId context_id_;
    uint32_t chip_id_ = 0;
    CoreCoord clock_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;

    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;

    // EMA of probe errors; resync stops early against this.
    std::chrono::nanoseconds typical_error_{};

    // High word only advances on low wrap (~3.2s at 1.35GHz).
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;
    std::atomic<int64_t> peak_probe_gap_ns_{0};

    ClockSyncMapping mapping_;
};

}  // namespace tt::tt_metal
