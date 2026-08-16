// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <utility>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

// Probe spacing for actively-drained devices. Certification needs two adjacent probe gaps plus
// read noise to fit inside kDvfsCertificateWindowBudget; the probe scheduler self-paces with
// tens of microseconds of wakeup jitter, so pairs land around 800 us with ~150 us of margin.
inline constexpr auto kDeviceClockSyncInterval = std::chrono::microseconds(375);

// Certification budget for the two-chord windows around a chord: the span under which at most
// one DVFS transition — start or glide tail — can touch the chord, so the neighbor chords are
// provably transition-free and their secants bracket every in-chord rate (monotonicity).
// Derivation, all terms verified against Blackhole firmware (tt-system-firmware dvfs.c,
// cm2dm_msg.c, clock_control_tt_bh.c) with Wormhole's 1 ms control loop confirmed by SysEng:
// AICLK moves once per the ARC firmware's 1 ms DVFS timer; a DMC board-power message can pull a
// pending tick 10 us early, and the DMC's own 1 ms send timer plus event-bit coalescing make
// even three pulls on one tick extreme (30 us); and a transition is a monotone PLL glide — the
// feedback divider walks one notch per ~100 ns write, ~3 us for a full swing on Blackhole,
// padded for Wormhole's unread glide mechanism (20 us; without this term two glides' tails
// could pincer a near-budget window from both sides). 1 ms − 30 us − 20 us. Host-forced clock
// operations (forced AICLK/VDD, AICLK sweep, clock-scheme switch) bypass the timer and void
// this constant.
inline constexpr auto kDvfsCertificateWindowBudget = std::chrono::microseconds(950);

// Maps device cycle-counter timestamps onto the host steady_clock timeline by interpolating
// between retained clock probes, publishing per record an affine mapping plus an error that
// upper-bounds the host-time placement error of both record endpoints.
//
// The error is a sound upper bound provided that: each probe's true read time lies within
// ±Anchor::error of its host_timestamp (bracketed read); the device counter is monotone and its
// rate piecewise-constant up to monotone transitions (DVFS glides) spaced and sized per
// kDvfsCertificateWindowBudget's derivation; and no host-forced clock operation is active.
//
// A chord (the interval between adjacent probes) is *certified* once the two-chord windows on
// both sides of it measure shorter than kDvfsCertificateWindowBudget: at most one transition can
// then touch the chord, and when one does no other transition touches either neighbor, so the
// neighbor secants bracket every rate inside the chord (monotonicity) and the worst-case
// interpolation error follows from that bracket — a hard bound under the assumptions above
// alone. Chords that cannot be certified (consumer stalls, history edges) and records whose
// start predates the retained history are bounded against the practical rate band instead —
// the spread of mature smoothed frequencies, noise-padded (the construction-time operating
// range until the first window matures). That tier adds one assumption: an unobserved window
// holds no rate the clock has never shown. It understates only when a first-ever rate
// excursion lands inside an unprobed gap; the alternative — the platform's full idle-to-limit
// range — never understates but overstates such records' error by ~50x.
//
// A chord's bound tightens when its successor probe lands (finalized_device_timestamp()); mapping
// a record whose end lies past that watermark is sound but yields the fallback-quality error.
//
// Not thread-safe: owned and driven by the consumer thread.
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

    // Default retained-probe count; the retained wall time is capacity times the probe interval.
    static constexpr size_t kProbeHistoryCapacity =
        std::bit_ceil(static_cast<size_t>(std::chrono::seconds(2) / kDeviceClockSyncInterval));

    // The initial practical rate band (the device's AICLK operating range) prices the fallback
    // tier until the first mature frequency window replaces — not folds into — it, so the
    // deliberately wide bring-up band cannot pin the spread wide forever.
    // `probe_history_capacity` (rounded up to a power of two) bounds retention: certified-tier
    // mapping must happen within capacity-times-interval of capture.
    explicit ClockSyncMapping(double rate_lo, double rate_hi, size_t probe_history_capacity = kProbeHistoryCapacity);

    // Requires probe host/device times strictly after the previous retained probe.
    void add_probe(const Anchor& probe);

    // Requires two retained probes: the constructor ingests the first, and the second exists
    // once a scheduler sweep of this device has been ingested — do not map before one
    // register-to-ingest round trip (~1.5 probe intervals). Certified-tier bounds also need
    // mapping within the retained history (kProbeHistoryCapacity probes, ~3 s at the default
    // cadence); records mapped later, or predating retention, ride from the oldest probe at the
    // practical rate band.
    [[nodiscard]] RecordMapping map_record(uint64_t start_device_timestamp, uint64_t end_device_timestamp);

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

    // Accepted probes discarded for breaking retained-timeline monotonicity (see add_probe).
    [[nodiscard]] uint64_t num_discarded_probes() const {
        return num_discarded_probes_.load(std::memory_order_relaxed);
    }

private:
    // Secant between adjacent probes, plus the allowance for a rate transition inside it.
    // Error quantities are precomputed integers: map_record runs per record on the consumer hot
    // path, so everything derivable per chord is derived here (at probe rate, not record rate).
    struct Chord {
        int64_t probe_error_ns = 0;  // max of the two probes' half-brackets
        // Worst-case in-chord misplacement from one transition: span * (sqrt(rho)-1)/(sqrt(rho)+1).
        int64_t nonlinearity_ns = 0;
        // Distance-to-nearest-probe refinement: below this many cycles, refine_slope * distance
        // beats nonlinearity_ns; zero when no rate bracket exists (refinement disabled).
        uint64_t refine_threshold_cycles = 0;
        double refine_slope = 0.0;  // ns per cycle: 1/rate_lo - 1/rate_hi
        // Nonzero exactly when this chord earned the certificate (a rate bracket exists).
        double rate_lo = 0.0;
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
        explicit SlidingExtremum(size_t capacity) : chord_index(capacity), value(capacity), mask(capacity - 1) {}

        std::vector<uint64_t> chord_index;
        std::vector<double> value;
        uint64_t mask;
        uint64_t head = 0;
        uint64_t tail = 0;

        [[nodiscard]] double front() const { return value[head & mask]; }
        void clear() { head = tail = 0; }
        // dominates(new_value, existing) == true drops the existing entry: <= for max-tracking,
        // >= for min-tracking.
        template <typename Dominates>
        void push(uint64_t index, double v, Dominates dominates) {
            while (head != tail && dominates(v, value[(tail - 1) & mask])) {
                --tail;
            }
            chord_index[tail & mask] = index;
            value[tail & mask] = v;
            ++tail;
        }
        // Drops entries whose chord lies at or before the window anchor.
        void evict_through(uint64_t index) {
            while (head != tail && chord_index[head & mask] <= index) {
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

    // Inverse-rate view of the practical band, for rides and overhang pricing.
    struct RideBand {
        double inv_mid = 0.0;
        double inv_half_spread = 0.0;
    };
    [[nodiscard]] RideBand ride_band() const;

    // One endpoint placed on the host timeline: on its chord's interpolation, or — when it
    // predates the retained history (a drain deeper than probe retention, i.e. a multi-second
    // consumer stall) — ridden back from the oldest probe at the practical band.
    struct EndpointPlacement {
        double host_ns = 0.0;
        int64_t error_ns = 0;
        const Chord* chord = nullptr;  // null: ridden (predates retained history)
    };
    [[nodiscard]] EndpointPlacement place_endpoint(uint64_t device_timestamp) const;

    // Extra bound for a timestamp past the chord's close (extrapolation beyond the newest probe:
    // holdback-evicted records during a deep probe outage). Mirrors the pre-history ride's
    // pricing: the gap between the extrapolation slope and any rate in the practical band grows
    // linearly with the overhang. Zero for in-chord timestamps and before a band exists.
    [[nodiscard]] int64_t forward_overhang_error_ns(const Chord& chord, uint64_t device_timestamp) const;

    void set_fallback_step_bound(Chord& chord, double span_ns) const;

    // Upgrades chord close_index from its fallback bound once both neighbors exist: evaluates the
    // certificate and computes the neighbor-bracket bound. Called when probe close_index+1 lands.
    void finalize_chord(uint64_t close_index);

    [[nodiscard]] uint64_t first_probe_at_or_past(uint64_t device_timestamp) const;

    [[nodiscard]] uint64_t oldest_probe() const {
        return probes_end_ > probe_capacity_ ? probes_end_ - probe_capacity_ : 0;
    }
    [[nodiscard]] const Anchor& probe_at(uint64_t index) const { return probe_history_[index & probe_mask_]; }
    [[nodiscard]] const Chord& chord_at(uint64_t close_index) const { return chords_[close_index & probe_mask_]; }

    const uint64_t probe_capacity_;
    const uint64_t probe_mask_;
    std::vector<Anchor> probe_history_;
    std::vector<Chord> chords_;  // chord ending at this probe index
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
    // this class is consumer-thread-only.
    std::atomic<uint64_t> num_finalized_chords_{0};
    std::atomic<uint64_t> num_certified_chords_{0};
    std::atomic<uint64_t> num_records_on_uncertified_chords_{0};
    std::atomic<uint64_t> num_discarded_probes_{0};

    // chord_index_around(start) cache, plus constants folded out of the per-record path (records
    // arrive chord-consecutive, and the validity check admits only finalized — immutable —
    // chords). offset = llround(offset_b * start + offset_a) equals the smoothed mapping up to
    // fp reordering (~1 ulp).
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
    bool active_finalized_ = false;
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
    // so contended-era brackets cannot poison it. Starts as the construction-time band; the
    // first mature window replaces it (provisional flag).
    double smoothed_frequency_min_ = 0.0;
    double smoothed_frequency_max_ = 0.0;
    bool practical_band_provisional_ = true;
};

// Reads a tensix free-running cycle counter over PCIe (bracketed between steady_clock reads) and
// broadcasts the probes to per-consumer Views, each owning its own ClockSyncMapping — minted
// like BroadcastRing readers, so any number of consumers ride one probe stream.
class DeviceClockSync {
public:
    using Anchor = ClockSyncMapping::Anchor;
    using RecordMapping = ClockSyncMapping::RecordMapping;

    static constexpr size_t kProbeHistoryCapacity = ClockSyncMapping::kProbeHistoryCapacity;

    // `clock_core` is a logical WORKER coordinate; any tensix works (a chip's wall clocks tick
    // together). The AICLK operating range is a bring-up requirement — query UMD's
    // get_min/max_clock_freq (MHz) and convert to GHz; it seeds every view's practical band and
    // the probe-plausibility margins, and wrong units silently reject every probe. Returns null
    // (after logging the reason) when the clock register cannot be mapped into a UC TLB window
    // or the first probe fails plausibility — a DeviceClockSync that exists can always be read.
    static std::unique_ptr<DeviceClockSync> create(
        ContextId context_id, IDevice* device, CoreCoord clock_core, double aiclk_min_ghz, double aiclk_max_ghz);
    ~DeviceClockSync();

    // Thread split: take_and_queue_probe() and the read-path state behind it belong to the
    // ProbeScheduler thread; each View belongs to one consumer thread. The probe ring is the
    // only bridge, and the atomic peak-gap counter the only other state both sides touch. The
    // constructor, which predates the scheduler, takes the first probe and stores it as every
    // future View's history seed.

    // Takes one bracketed clock probe (best of up to kResyncProbes reads) and queues it for the
    // views; a read failing the plausibility check below even after a hardware re-read is
    // skipped, and the views simply see a longer chord. Scheduler thread only.
    void take_and_queue_probe();

    // Per-consumer handle, minted from the owner like a BroadcastRing reader: it owns its probe
    // reader and its ClockSyncMapping, seeded with the owner's construction-time probe so no
    // record this device produces can predate the view's history — a view created long after
    // bring-up simply starts with one long fallback-priced chord. One consumer thread per view.
    class View {
    public:
        // `probe_history_capacity` bounds this view's retention (see ClockSyncMapping); size it
        // so capacity-times-interval covers the consumer's capture-to-map latency. The sync must
        // outlive the view.
        explicit View(DeviceClockSync& sync, size_t probe_history_capacity = ClockSyncMapping::kProbeHistoryCapacity);

        View(const View&) = delete;
        View& operator=(const View&) = delete;
        View(View&&) = delete;
        View& operator=(View&&) = delete;

        // Drains queued probes into the mapping. Call at least once per kProbeRingCapacity
        // probe intervals (~3 s at kDeviceClockSyncInterval) — a lapped ring costs the gap's
        // chords their certificates, exactly as a consumer stall would.
        void ingest_queued_probes();

        // See ClockSyncMapping::map_record(); the two-probe warm-up is one register-to-ingest
        // round trip after the owner joins a scheduler (~1.5 probe intervals).
        [[nodiscard]] RecordMapping map_record(uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
            return mapping_.map_record(start_device_timestamp, end_device_timestamp);
        }

        // See ClockSyncMapping::finalized_device_timestamp().
        [[nodiscard]] uint64_t finalized_device_timestamp() const { return mapping_.finalized_device_timestamp(); }

        [[nodiscard]] uint64_t num_finalized_chords() const { return mapping_.num_finalized_chords(); }
        [[nodiscard]] uint64_t num_certified_chords() const { return mapping_.num_certified_chords(); }
        [[nodiscard]] uint64_t num_records_on_uncertified_chords() const {
            return mapping_.num_records_on_uncertified_chords();
        }
        // Probes discarded as non-monotone (see ClockSyncMapping::add_probe).
        [[nodiscard]] uint64_t num_discarded_probes() const { return mapping_.num_discarded_probes(); }

    private:
        BroadcastRing<Anchor>::Reader reader_;
        ClockSyncMapping mapping_;
    };

    // True when the rate implied between two probes lies inside [rate_lo, rate_hi] (cycles/ns).
    // A corrupted PCIe read moves the 64-bit timestamp by ~2^32 cycles (~11,000 GHz over one
    // probe interval) or freezes/reverses it, so any envelope near the silicon's range separates
    // garbage from every legitimate reading, DVFS steps and multi-second gaps included.
    [[nodiscard]] static bool plausible_probe_step(
        const Anchor& previous, const Anchor& next, double rate_lo, double rate_hi);

    // Implausible reads rejected at the read path; each costs one probe cycle.
    [[nodiscard]] uint64_t num_rejected_probes() const { return num_rejected_probes_.load(std::memory_order_relaxed); }

    // Largest gap between consecutive probes since the last call; reading clears it. Callable from
    // any thread.
    [[nodiscard]] std::chrono::nanoseconds take_peak_probe_gap() {
        return std::chrono::nanoseconds(peak_probe_gap_ns_.exchange(0, std::memory_order_relaxed));
    }

private:
    void configure_clock_read_path();

    // Nullopt when the read fails plausibility even after the hardware re-read.
    DeviceClockSync(
        ContextId context_id, IDevice* device, CoreCoord clock_core, double aiclk_min_ghz, double aiclk_max_ghz);

    [[nodiscard]] std::optional<Anchor> read_probe();

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

    // Plausibility envelope for the rate between consecutive accepted probes, margined from the
    // AICLK operating range in the constructor.
    double plausible_rate_lo_;
    double plausible_rate_hi_;
    Anchor last_accepted_{};
    bool has_last_accepted_ = false;
    bool force_hi_read_ = false;
    int consecutive_rejections_ = 0;
    std::chrono::steady_clock::time_point last_reject_warn_;
    std::atomic<uint64_t> num_rejected_probes_{0};

    // Probe pipe: the scheduler thread writes each probe it takes, each View drains them into
    // its mapping at its own pace. Sized so the writer laps a reader only after that consumer
    // has been absent for multiple seconds; a lap costs the gap's chords their certificates,
    // exactly as the stall itself would.
    static constexpr size_t kProbeRingCapacity = 8192;  // ~3 s of cadence probes of consumer absence
    BroadcastRing<Anchor> probe_ring_{kProbeRingCapacity};

    // Every View's history seed; taken at construction, before the device can produce a record.
    std::optional<Anchor> first_probe_;
    double aiclk_min_ghz_ = 0.0;
    double aiclk_max_ghz_ = 0.0;
};

// Sole owner of a probe cadence: one thread sweeps every registered device's clock at the
// chosen interval, from the moment the device is constructed until the scheduler is destroyed
// (stop-and-join), queueing each probe on the device's clock sync for its consumer. There is
// deliberately no handoff anywhere in a consumer's life — a single continuous schedule is what
// keeps consumer construction, however long, from becoming a blind multi-ms chord.
//
// The interval buys the error tier: certification needs adjacent probe-gap pairs (two intervals
// plus jitter) inside kDvfsCertificateWindowBudget, so intervals up to kDeviceClockSyncInterval
// earn certified (~us) bounds, while slower cadences are legal but price every record at the
// fallback tier (band-quality bounds) for a proportionally smaller probe load. Retention spans
// scale with it too: the mapping's probe history and each device's probe ring hold a fixed
// probe count, so their wall-clock reach is proportional to the interval.
class ProbeScheduler {
public:
    explicit ProbeScheduler(std::chrono::nanoseconds probe_interval);
    ~ProbeScheduler();

    ProbeScheduler(const ProbeScheduler&) = delete;
    ProbeScheduler& operator=(const ProbeScheduler&) = delete;
    ProbeScheduler(ProbeScheduler&&) = delete;
    ProbeScheduler& operator=(ProbeScheduler&&) = delete;

    // Called from the initializing thread as each device comes up. The sync must outlive this
    // scheduler (no unregister — destroy the scheduler first). The device joins the sweep half
    // an interval out: its constructor already probed, and an immediate probe would mint a tiny
    // chord whose noisy secant poisons its neighbors' certified rate brackets.
    void register_device(DeviceClockSync& clock_sync);

    // The scheduler sweeps only while at least one Demand token is alive; with none, the
    // thread keeps its cadence but skips the probes. A token says "someone will consume these
    // probes" — hold one for exactly that scope, and pausing needs no undo anywhere else.
    class Demand {
    public:
        explicit Demand(ProbeScheduler& scheduler) : scheduler_(&scheduler) {
            scheduler_->demand_.fetch_add(1, std::memory_order_relaxed);
        }
        ~Demand() {
            if (scheduler_ != nullptr) {
                scheduler_->demand_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        Demand(Demand&& other) noexcept : scheduler_(std::exchange(other.scheduler_, nullptr)) {}
        Demand(const Demand&) = delete;
        Demand& operator=(const Demand&) = delete;
        Demand& operator=(Demand&&) = delete;

    private:
        ProbeScheduler* scheduler_;
    };
    [[nodiscard]] Demand demand() { return Demand(*this); }

private:
    struct Entry {
        DeviceClockSync* clock_sync = nullptr;
        std::chrono::steady_clock::time_point earliest_allowed;
    };
    void run(const std::stop_token& stop);

    const std::chrono::nanoseconds probe_interval_;
    std::atomic<int> demand_{0};
    std::mutex mutex_;
    std::vector<Entry> entries_;
    std::jthread thread_;
};

}  // namespace tt::tt_metal
