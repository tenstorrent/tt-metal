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

// Probe spacing for actively-drained devices. Certification needs two adjacent probe gaps plus
// read noise to fit inside kDvfsMinTransitionSpacing; the sync thread self-paces with tens of
// microseconds of wakeup jitter, so pairs land around 800 us with ~100 us of margin.
inline constexpr auto kDeviceClockSyncInterval = std::chrono::microseconds(375);

// Minimum spacing between starts of device clock-rate transitions. AICLK is stepped by the ARC
// firmware's 1 ms DVFS timer, one monotone PLL glide per tick; a board-power message can pull a
// tick up to ~10 us early, and the 50 us margin absorbs several such pulls stacking on one tick.
// Verified against Blackhole firmware (tt-system-firmware dvfs.c/pll driver) and confirmed by
// SysEng for Wormhole (same 1 ms ARC control-loop period on both). Host-forced clock operations
// (forced AICLK/VDD, AICLK sweep, clock-scheme switch) bypass the timer and void this constant.
inline constexpr auto kDvfsMinTransitionSpacing = std::chrono::microseconds(950);

// Maps device cycle-counter timestamps onto the host steady_clock timeline by interpolating
// between retained clock probes, publishing per record an affine mapping plus an error that
// upper-bounds the host-time placement error of both record endpoints.
//
// The error is a sound upper bound provided that: each probe's true read time lies within
// ±Anchor::error of its host_timestamp (bracketed read); the device counter is monotone and its
// rate piecewise-constant up to monotone transitions (DVFS glides) whose starts are at least
// kDvfsMinTransitionSpacing apart; and no host-forced clock operation is active.
//
// A chord (the interval between adjacent probes) is *certified* once the two-chord windows on
// both sides of it measure shorter than kDvfsMinTransitionSpacing: at most one transition can
// then touch the chord, its neighbors are transition-free, their secant rates bracket every rate
// inside the chord, and the worst-case interpolation error follows from that bracket — a hard
// bound under the assumptions above alone. Chords that cannot be certified (receiver stalls,
// history edges) and records whose start predates the retained history are bounded against the
// practical rate band instead — the spread of mature smoothed frequencies, noise-padded (the
// chord's own span until a window has matured). That tier adds one assumption: an unobserved
// window holds no rate the clock has never shown. It understates only when a first-ever rate
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

    static constexpr size_t kProbeHistoryCapacity =
        std::bit_ceil(static_cast<size_t>(std::chrono::seconds(2) / kDeviceClockSyncInterval));

    // Requires probe host/device times strictly after the previous retained probe.
    void add_probe(const Anchor& probe);

    // Nullopt only while fewer than two probes are retained (no rate knowledge yet); records
    // predating the retained history ride back from its oldest probe at the practical rate band.
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

    // Records whose bound came from an uncertified chord or the pre-history ride. They are still
    // published, with fallback-tier bounds — this only counts them, so callers can assert the
    // tier stays a tiny fraction of traffic.
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
        double smoothed_frequency = 0.0;  // frequency-window regression as of finalize; what records publish
        // |1/smoothed_frequency - 1/frequency|, ns per cycle: consumer host_end reconstruction skew.
        double smoothing_skew_per_cycle = 0.0;
        // Largest skew read noise alone can put on this chord's secant; skew beyond it is
        // evidence of a transition inside the chord.
        double noise_skew_per_cycle = 0.0;
        uint64_t open_device_timestamp = 0;
        double open_host_ns = 0.0;
        uint64_t close_device_timestamp = 0;
    };

    // Baseline for the published frequency: at this span read noise costs the regression well
    // under a ppm, and the anchor stays safely inside the 2 s probe retention. The window slides
    // continuously, so there is no re-anchor burst.
    static constexpr auto kFrequencyWindowMax = std::chrono::milliseconds(1600);

    // Sliding-window extremum (monotonic deque): each entry enters and leaves once — O(1)
    // amortized push/evict, no rescan when the anchor passes entries holding the extremum.
    struct SlidingExtremum {
        std::array<uint64_t, kProbeHistoryCapacity> chord_index{};
        std::array<double, kProbeHistoryCapacity> value{};
        uint64_t head = 0;
        uint64_t tail = 0;

        [[nodiscard]] bool empty() const { return head == tail; }
        [[nodiscard]] double front() const { return value[head & (kProbeHistoryCapacity - 1)]; }
        void clear() { head = tail = 0; }
        // dominates(new_value, existing) == true drops the existing entry: <= for max-tracking,
        // >= for min-tracking.
        template <typename Dominates>
        void push(uint64_t index, double v, Dominates dominates) {
            while (head != tail && dominates(v, value[(tail - 1) & (kProbeHistoryCapacity - 1)])) {
                --tail;
            }
            chord_index[tail & (kProbeHistoryCapacity - 1)] = index;
            value[tail & (kProbeHistoryCapacity - 1)] = v;
            ++tail;
        }
        // Drops entries whose chord lies at or before the window anchor.
        void evict_through(uint64_t index) {
            while (head != tail && chord_index[head & (kProbeHistoryCapacity - 1)] <= index) {
                ++head;
            }
        }
    };

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

    // Least squares over the window's probes via running sums: O(1) add/remove, O(1) origin
    // rebase (shift identities keep magnitudes at window scale). ~sqrt(N/12) tighter than the
    // endpoint secant: ~0.02 ppm at a mature window.
    struct WindowRegression {
        double n = 0;
        double sum_t = 0;
        double sum_d = 0;
        double sum_tt = 0;
        double sum_td = 0;
        double origin_host_ns = 0;
        double origin_device = 0;

        void clear(double host_ns, double device) {
            *this = WindowRegression{};
            origin_host_ns = host_ns;
            origin_device = device;
        }
        void add(double host_ns, double device) {
            const double t = host_ns - origin_host_ns;
            const double d = device - origin_device;
            n += 1;
            sum_t += t;
            sum_d += d;
            sum_tt += t * t;
            sum_td += t * d;
        }
        void remove(double host_ns, double device) {
            const double t = host_ns - origin_host_ns;
            const double d = device - origin_device;
            n -= 1;
            sum_t -= t;
            sum_d -= d;
            sum_tt -= t * t;
            sum_td -= t * d;
        }
        void rebase(double host_ns, double device) {
            const double dt = host_ns - origin_host_ns;
            const double dd = device - origin_device;
            sum_tt += -2.0 * dt * sum_t + n * dt * dt;
            sum_td += -dd * sum_t - dt * sum_d + n * dt * dd;
            sum_t -= n * dt;
            sum_d -= n * dd;
            origin_host_ns = host_ns;
            origin_device = device;
        }
        // Ticks per host ns; NaN-free fallback is the caller's endpoint secant.
        [[nodiscard]] double slope() const {
            const double det = n * sum_tt - sum_t * sum_t;
            return det > 0.0 ? (n * sum_td - sum_t * sum_d) / det : 0.0;
        }
    };

    // Frequency window: the run of consecutive certified chords whose rate brackets all intersect.
    // Maintaining the *running intersection* (not pairwise checks) makes hidden-step creep
    // impossible: every rate in the window provably lies inside it; only a detected transition
    // (empty intersection) resets it.
    bool frequency_window_active_ = false;
    uint64_t frequency_window_anchor_ = 0;  // probe index the window is measured from
    SlidingExtremum window_rate_lo_;        // max of in-window rate_lo
    SlidingExtremum window_rate_hi_;        // min of in-window rate_hi
    WindowRegression window_regression_;    // over probes [anchor, close]

    // Relaxed atomics so diagnostics accessors are readable from any thread; everything else on
    // this class is receiver-thread-only.
    std::atomic<uint64_t> num_finalized_chords_{0};
    std::atomic<uint64_t> num_certified_chords_{0};
    std::atomic<uint64_t> num_records_on_uncertified_chords_{0};

    // chord_index_around(start) cache, plus constants folded out of the per-record path (records
    // arrive chord-consecutive; holdback maps only finalized — immutable — chords, so the cache
    // cannot go stale). offset = llround(offset_b * start + offset_a) equals the smoothed
    // mapping up to fp reordering (~1 ulp).
    struct ActiveChordConstants {
        uint64_t open_device_timestamp = 0;
        uint64_t close_device_timestamp = 0;
        double offset_a = 0.0;
        double offset_b = 0.0;
        int64_t base_error_ns = 0;  // probe_error + nonlinearity: the mid-chord bound
        uint64_t refine_threshold_cycles = 0;
        double refine_slope = 0.0;
        int64_t probe_error_ns = 0;
        double smoothing_skew_per_cycle = 0.0;
        double smoothed_frequency = 0.0;
        double frequency = 0.0;      // chord secant
        double secant_offset = 0.0;  // open_device - frequency * open_host_ns
        bool transition_evident = false;
        bool certified = false;
    };
    std::optional<uint64_t> active_chord_index_;
    ActiveChordConstants active_;

    void refresh_active_chord_constants();

    // error_ns_on specialized to in-bounds timestamps on the active chord.
    [[nodiscard]] int64_t active_error_ns(uint64_t device_timestamp) const {
        const uint64_t distance_cycles = std::min(
            device_timestamp - active_.open_device_timestamp, active_.close_device_timestamp - device_timestamp);
        if (distance_cycles >= active_.refine_threshold_cycles) {
            return active_.base_error_ns;
        }
        return active_.probe_error_ns +
               static_cast<int64_t>(active_.refine_slope * static_cast<double>(distance_cycles)) + 1;
    }

    // Spread of mature smoothed frequencies: prices the pre-history ride and, noise-padded, the
    // uncertified-chord fallback. Grows only from certified chords' ppm-accurate window values,
    // so contended-era brackets cannot poison it.
    double smoothed_frequency_min_ = 0.0;
    double smoothed_frequency_max_ = 0.0;
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

    // Thread split: read_probe() and the read-path state behind it belong to the sync thread;
    // ingest_probe(), map_record(), and the mapping belong to the receiver thread. The atomic
    // peak-gap counter is the only state both sides touch.

    // Takes one bracketed clock probe (best of up to kResyncProbes reads). Sync thread only.
    // Nullopt when the read fails the plausibility check below even after a hardware re-read:
    // the cycle is skipped and the mapping simply sees a longer chord.
    [[nodiscard]] std::optional<Anchor> read_probe();

    // True when the rate implied between two probes lies inside [rate_lo, rate_hi] (cycles/ns).
    // A corrupted PCIe read moves the 64-bit timestamp by ~2^32 cycles (~11,000 GHz over one
    // probe interval) or freezes/reverses it, so any envelope near the silicon's range separates
    // garbage from every legitimate reading, DVFS steps and multi-second gaps included.
    [[nodiscard]] static bool plausible_probe_step(
        const Anchor& previous, const Anchor& next, double rate_lo, double rate_hi);

    [[nodiscard]] uint64_t num_rejected_probes() const { return num_rejected_probes_.load(std::memory_order_relaxed); }

    // Feeds a probe from read_probe() into the mapping. Receiver thread only; probes must arrive
    // in the order they were taken.
    void ingest_probe(const Anchor& probe) { mapping_.add_probe(probe); }

    // See ClockSyncMapping::map_record().
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
    void configure_plausible_rate_band();

    Anchor probe(bool force_read_hi);

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

    // Plausibility envelope for the rate between consecutive accepted probes; defaults hold when
    // the AICLK range query is unavailable (mock/simulated paths).
    double plausible_rate_lo_ = 0.05;
    double plausible_rate_hi_ = 8.0;
    Anchor last_accepted_{};
    bool has_last_accepted_ = false;
    bool force_hi_read_ = false;
    int consecutive_rejections_ = 0;
    std::chrono::steady_clock::time_point last_reject_warn_;
    std::atomic<uint64_t> num_rejected_probes_{0};

    ClockSyncMapping mapping_;
};

}  // namespace tt::tt_metal
