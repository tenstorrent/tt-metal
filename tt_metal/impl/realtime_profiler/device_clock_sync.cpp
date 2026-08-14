// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/device_clock_sync.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/device.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/tlb.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

constexpr auto kMaxProbeGapBeforeRereadingHi = std::chrono::seconds(1);

namespace {

double ns_count(std::chrono::nanoseconds duration) { return static_cast<double>(duration.count()); }

double host_ns(const ClockSyncMapping::Anchor& anchor) {
    return static_cast<double>(anchor.host_timestamp.time_since_epoch().count());
}

double span_ns_between(const ClockSyncMapping::Anchor& open, const ClockSyncMapping::Anchor& close) {
    return ns_count(std::chrono::duration_cast<std::chrono::nanoseconds>(close.host_timestamp - open.host_timestamp));
}

// Widens [lo, hi] to contain the true rate behind a measured secant: the chord's host span is
// known only to within the two probes' read errors, so the true rate lies in
// [frequency * span/(span+slack), frequency * span/(span-slack)] ⊆ [frequency*(1-u), frequency*(1+u)].
bool widen_rate_bracket(
    const ClockSyncMapping::Anchor& open,
    const ClockSyncMapping::Anchor& close,
    double frequency,
    double& lo,
    double& hi) {
    const double span = span_ns_between(open, close);
    const double slack = ns_count(open.error + close.error);
    if (span <= slack || frequency <= 0.0) {
        return false;
    }
    const double u = slack / (span - slack);
    lo = std::min(lo, frequency * (1.0 - u));
    hi = std::max(hi, frequency * (1.0 + u));
    return lo > 0.0;
}

// Worst-case misplacement of linear interpolation across one rate transition inside a window of
// span_ns, with the rates on both sides confined to a ratio of rho: span * (sqrt(rho)-1)/(sqrt(rho)+1).
int64_t transition_step_bound_ns(double span_ns, double rho) {
    const double sqrt_rho = std::sqrt(rho);
    return std::llround(span_ns * (sqrt_rho - 1.0) / (sqrt_rho + 1.0));
}

uint64_t refine_threshold_for(int64_t nonlinearity_ns, double slope) {
    return slope > 0.0 ? static_cast<uint64_t>(static_cast<double>(nonlinearity_ns) / slope) : 0;
}

}  // namespace

// The band stores smoothed, ppm-accurate frequencies, but a blind window's instantaneous rate
// can sit slightly outside them without being a new operating point (short-term wobble a mature
// window averages through). Pad by the ~0.4% rate uncertainty read noise puts on a single
// chord, rounded up.
constexpr double kFallbackBandNoisePad = 0.005;

void ClockSyncMapping::set_fallback_step_bound(Chord& chord, double span_ns) const {
    // The mature smoothed band, not a lifetime fold of raw secants — that fold creeps wider with
    // every extreme noise draw and contended-era brackets poison it for good.
    if (!(smoothed_frequency_min_ > 0.0)) {
        // No rate knowledge yet: containment of both true and interpolated time in the chord,
        // with no distance refinement (threshold zero).
        chord.nonlinearity_ns = std::llround(span_ns);
        chord.refine_slope = 0.0;
        chord.refine_threshold_cycles = 0;
        return;
    }
    const double lo = smoothed_frequency_min_ * (1.0 - kFallbackBandNoisePad);
    const double hi = smoothed_frequency_max_ * (1.0 + kFallbackBandNoisePad);
    chord.nonlinearity_ns = transition_step_bound_ns(span_ns, hi / lo);
    chord.refine_slope = 1.0 / lo - 1.0 / hi;
    chord.refine_threshold_cycles = refine_threshold_for(chord.nonlinearity_ns, chord.refine_slope);
}

void ClockSyncMapping::add_probe(const Anchor& probe) {
    const uint64_t close_index = probes_end_;
    const size_t slot = close_index & (kProbeHistoryCapacity - 1);
    probe_history_[slot] = probe;
    ++probes_end_;

    if (probes_end_ - oldest_probe() < 2) {
        return;
    }
    const Anchor& open = probe_at(close_index - 1);
    const double span_ns = span_ns_between(open, probe);
    const double frequency = static_cast<double>(probe.device_timestamp - open.device_timestamp) / span_ns;
    const double slack_ns = ns_count(open.error + probe.error);
    const double u = slack_ns < span_ns ? slack_ns / (span_ns - slack_ns) : 1.0;
    Chord& chord = chords_[slot];
    chord = Chord{
        .probe_error_ns = std::max(open.error, probe.error).count(),
        .frequency = frequency,
        .smoothed_frequency = frequency,
        .noise_skew_per_cycle = u < 1.0 ? 1.0 / (frequency * (1.0 - u)) - 1.0 / (frequency * (1.0 + u))
                                        : std::numeric_limits<double>::infinity(),
        .open_device_timestamp = open.device_timestamp,
        .open_host_ns = host_ns(open),
        .close_device_timestamp = probe.device_timestamp,
    };
    set_fallback_step_bound(chord, span_ns + ns_count(open.error + probe.error));
    finalize_chord(close_index - 1);
}

void ClockSyncMapping::finalize_chord(uint64_t close_index) {
    const uint64_t begin = oldest_probe();
    if (close_index < begin + 2 || close_index + 1 >= probes_end_) {
        return;
    }
    const Anchor& outer_open = probe_at(close_index - 2);
    const Anchor& open = probe_at(close_index - 1);
    const Anchor& close = probe_at(close_index);
    const Anchor& outer_close = probe_at(close_index + 1);
    Chord& chord = chords_[close_index & (kProbeHistoryCapacity - 1)];
    num_finalized_chords_.fetch_add(1, std::memory_order_relaxed);
    const double span_ns = span_ns_between(open, close) + ns_count(open.error + close.error);

    double rate_lo = std::numeric_limits<double>::infinity();
    double rate_hi = 0.0;
    if (!widen_rate_bracket(outer_open, open, chord_at(close_index - 1).frequency, rate_lo, rate_hi) ||
        !widen_rate_bracket(close, outer_close, chord_at(close_index + 1).frequency, rate_lo, rate_hi)) {
        frequency_window_active_ = false;
        set_fallback_step_bound(chord, span_ns);
        return;
    }

    const auto back_window = (close.host_timestamp - outer_open.host_timestamp) + outer_open.error + close.error;
    const auto forward_window = (outer_close.host_timestamp - open.host_timestamp) + open.error + outer_close.error;
    if (back_window >= kDvfsMinTransitionSpacing || forward_window >= kDvfsMinTransitionSpacing) {
        frequency_window_active_ = false;
        // Holdback publishes records only after finalize, so this restamp — not the creation-time
        // one — is the bound they see.
        set_fallback_step_bound(chord, span_ns);
        return;
    }

    chord.nonlinearity_ns = transition_step_bound_ns(span_ns, rate_hi / rate_lo);
    chord.refine_slope = 1.0 / rate_lo - 1.0 / rate_hi;
    chord.refine_threshold_cycles = refine_threshold_for(chord.nonlinearity_ns, chord.refine_slope);
    chord.rate_lo = rate_lo;
    chord.rate_hi = rate_hi;
    num_certified_chords_.fetch_add(1, std::memory_order_relaxed);

    const auto push_bracket = [&] {
        window_rate_lo_.push(close_index, rate_lo, [](double v, double back) { return back <= v; });
        window_rate_hi_.push(close_index, rate_hi, [](double v, double back) { return back >= v; });
    };
    const auto restart_window = [&] {
        frequency_window_anchor_ = close_index - 1;
        window_rate_lo_.clear();
        window_rate_hi_.clear();
        push_bracket();
        window_regression_.clear(host_ns(open), static_cast<double>(open.device_timestamp));
        window_regression_.add(host_ns(open), static_cast<double>(open.device_timestamp));
        window_regression_.add(host_ns(close), static_cast<double>(close.device_timestamp));
    };
    if (!frequency_window_active_ || frequency_window_anchor_ < begin) {
        frequency_window_active_ = true;
        restart_window();
    } else {
        push_bracket();
        window_regression_.add(host_ns(close), static_cast<double>(close.device_timestamp));
        while (close.host_timestamp - probe_at(frequency_window_anchor_).host_timestamp > kFrequencyWindowMax) {
            const Anchor& evicted = probe_at(frequency_window_anchor_);
            window_regression_.remove(host_ns(evicted), static_cast<double>(evicted.device_timestamp));
            ++frequency_window_anchor_;
        }
        window_rate_lo_.evict_through(frequency_window_anchor_);
        window_rate_hi_.evict_through(frequency_window_anchor_);
        const Anchor& rebased = probe_at(frequency_window_anchor_);
        window_regression_.rebase(host_ns(rebased), static_cast<double>(rebased.device_timestamp));
        if (window_rate_lo_.front() > window_rate_hi_.front()) {
            // Empty intersection: a detected transition; the window restarts at this chord.
            restart_window();
        }
    }
    const Anchor& anchor = probe_at(frequency_window_anchor_);
    const double window_span_ns = span_ns_between(anchor, close);
    const double regression_slope = window_regression_.slope();
    chord.smoothed_frequency =
        regression_slope > 0.0 ? regression_slope
                               : static_cast<double>(close.device_timestamp - anchor.device_timestamp) / window_span_ns;
    chord.smoothing_skew_per_cycle = std::abs(1.0 / chord.smoothed_frequency - 1.0 / chord.frequency);
    // Young windows would widen the practical spread with plain secant noise; fold only once the
    // baseline makes the smoothed value ppm-accurate.
    if (window_span_ns >= ns_count(std::chrono::duration_cast<std::chrono::nanoseconds>(kFrequencyWindowMax)) / 2) {
        if (smoothed_frequency_max_ == 0.0) {
            smoothed_frequency_min_ = chord.smoothed_frequency;
            smoothed_frequency_max_ = chord.smoothed_frequency;
        } else {
            smoothed_frequency_min_ = std::min(smoothed_frequency_min_, chord.smoothed_frequency);
            smoothed_frequency_max_ = std::max(smoothed_frequency_max_, chord.smoothed_frequency);
        }
    }
}

int64_t ClockSyncMapping::error_ns_on(const Chord& chord, uint64_t device_timestamp) {
    if (device_timestamp < chord.open_device_timestamp || device_timestamp > chord.close_device_timestamp) {
        return chord.probe_error_ns + chord.nonlinearity_ns;
    }
    // The true and interpolated maps agree at the probes and their per-cycle slopes both lie in
    // [1/f_hi, 1/f_lo], so their difference grows from either probe at most at the slope width.
    // Past the precomputed threshold the refinement cannot beat nonlinearity_ns, so the common
    // (mid-chord) case is integer-only.
    const uint64_t distance_cycles =
        std::min(device_timestamp - chord.open_device_timestamp, chord.close_device_timestamp - device_timestamp);
    if (distance_cycles >= chord.refine_threshold_cycles) {
        return chord.probe_error_ns + chord.nonlinearity_ns;
    }
    return chord.probe_error_ns + static_cast<int64_t>(chord.refine_slope * static_cast<double>(distance_cycles)) + 1;
}

uint64_t ClockSyncMapping::first_probe_at_or_past(uint64_t device_timestamp) const {
    const uint64_t begin = oldest_probe();
    const uint64_t end = probes_end_;

    uint64_t i = last_probe_index_;
    if (i < begin || i >= end) {
        i = begin;
    }
    if (probe_at(i).device_timestamp < device_timestamp) {
        while (i < end && probe_at(i).device_timestamp < device_timestamp) {
            ++i;
        }
    } else {
        while (i > begin && probe_at(i - 1).device_timestamp >= device_timestamp) {
            --i;
        }
    }
    last_probe_index_ = i < end ? i : end - 1;
    return i;
}

std::optional<uint64_t> ClockSyncMapping::chord_index_around(uint64_t device_timestamp) const {
    const uint64_t begin = oldest_probe();
    if (probes_end_ - begin < 2) {
        return std::nullopt;
    }
    const uint64_t close_index = std::min(first_probe_at_or_past(device_timestamp), probes_end_ - 1);
    if (close_index == begin) {
        return std::nullopt;
    }
    return close_index;
}

void ClockSyncMapping::refresh_active_chord_constants() {
    const Chord& chord = chord_at(*active_chord_index_);
    const double smoothed = chord.smoothed_frequency;
    active_.open_device_timestamp = chord.open_device_timestamp;
    active_.close_device_timestamp = chord.close_device_timestamp;
    active_.offset_b = 1.0 - smoothed / chord.frequency;
    active_.offset_a =
        smoothed * (static_cast<double>(chord.open_device_timestamp) / chord.frequency - chord.open_host_ns);
    active_.base_error_ns = chord.probe_error_ns + chord.nonlinearity_ns;
    active_.refine_threshold_cycles = chord.refine_threshold_cycles;
    active_.refine_slope = chord.refine_slope;
    active_.probe_error_ns = chord.probe_error_ns;
    active_.smoothing_skew_per_cycle = chord.smoothing_skew_per_cycle;
    active_.smoothed_frequency = smoothed;
    active_.frequency = chord.frequency;
    active_.secant_offset = static_cast<double>(chord.open_device_timestamp) - chord.frequency * chord.open_host_ns;
    active_.transition_evident = chord.smoothing_skew_per_cycle > chord.noise_skew_per_cycle;
    active_.certified = chord.rate_lo != 0.0;
}

std::optional<ClockSyncMapping::RecordMapping> ClockSyncMapping::map_record(
    uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
    const uint64_t begin = oldest_probe();
    const bool active_chord_valid = active_chord_index_.has_value() && *active_chord_index_ > begin &&
                                    *active_chord_index_ < probes_end_ &&
                                    start_device_timestamp > active_.open_device_timestamp &&
                                    start_device_timestamp <= active_.close_device_timestamp;
    if (!active_chord_valid) {
        active_chord_index_ = chord_index_around(start_device_timestamp);
        if (active_chord_index_.has_value()) {
            refresh_active_chord_constants();
        }
    }

    if (active_chord_index_.has_value() && end_device_timestamp <= active_.close_device_timestamp) {
        if (!active_.certified) {
            num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
        }
        const int64_t skew_term_ns = active_.smoothing_skew_per_cycle != 0.0
                                         ? static_cast<int64_t>(
                                               static_cast<double>(end_device_timestamp - start_device_timestamp) *
                                               active_.smoothing_skew_per_cycle) +
                                               1
                                         : 0;
        const uint64_t start_distance = std::min(
            start_device_timestamp - active_.open_device_timestamp,
            active_.close_device_timestamp - start_device_timestamp);
        const uint64_t end_distance = std::min(
            end_device_timestamp - active_.open_device_timestamp,
            active_.close_device_timestamp - end_device_timestamp);
        int64_t start_error_ns;
        int64_t end_error_ns;
        if (start_distance >= active_.refine_threshold_cycles && end_distance >= active_.refine_threshold_cycles) {
            start_error_ns = active_.base_error_ns;
            end_error_ns = active_.base_error_ns;
        } else {
            start_error_ns = active_error_ns(start_device_timestamp);
            end_error_ns = active_error_ns(end_device_timestamp);
        }
        // Same arbitration as the spanning path, gated on skew beyond what read noise can put on
        // the secant: only a transition inside the chord separates it further from the window
        // value, and there the secant — the chord's true average rate up to endpoint noise — is
        // nearer every record's average rate. The gate keeps noisy quiet-chord secants from
        // displacing the regression on near-chord-span records.
        if (active_.transition_evident && skew_term_ns > std::max(start_error_ns, end_error_ns)) {
            return RecordMapping{
                .device_cycle_offset = std::llround(active_.secant_offset),
                .error = std::chrono::nanoseconds(std::max(start_error_ns, end_error_ns)),
                .frequency = active_.frequency};
        }
        return RecordMapping{
            .device_cycle_offset =
                std::llround(active_.offset_b * static_cast<double>(start_device_timestamp) + active_.offset_a),
            .error = std::chrono::nanoseconds(std::max(start_error_ns, end_error_ns + skew_term_ns)),
            .frequency = active_.smoothed_frequency};
    }

    RecordMapping mapping;
    if (!active_chord_index_.has_value()) {
        // Start is older than our probe history.
        const std::optional<uint64_t> end_index = chord_index_around(end_device_timestamp);
        if (!end_index.has_value()) {
            return std::nullopt;
        }
        const Chord& end_chord = chord_at(*end_index);
        const double end_host_ns = host_ns_on(end_chord, end_device_timestamp);
        const int64_t end_error_ns = error_ns_on(end_chord, end_device_timestamp);
        // Ride back from the oldest retained probe at the mature smoothed-frequency spread (a few
        // ppm on a stable clock), so a long program claims microseconds of start error rather
        // than a per-chord noise band's milliseconds.
        num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
        double ride_lo = smoothed_frequency_min_;
        double ride_hi = smoothed_frequency_max_;
        if (!(ride_lo > 0.0)) {
            ride_lo = end_chord.frequency;
            ride_hi = end_chord.frequency;
        }
        const Anchor& ring_oldest = probe_at(begin);
        const double uncovered_cycles = static_cast<double>(ring_oldest.device_timestamp - start_device_timestamp);
        const double start_host_ns = host_ns(ring_oldest) - uncovered_cycles * (1.0 / ride_lo + 1.0 / ride_hi) * 0.5;
        const int64_t start_error_ns =
            std::llround(uncovered_cycles * (1.0 / ride_lo - 1.0 / ride_hi) * 0.5) + ring_oldest.error.count();
        const double frequency =
            static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
        mapping = RecordMapping{
            .device_cycle_offset = std::llround(static_cast<double>(end_device_timestamp) - frequency * end_host_ns),
            .error = std::chrono::nanoseconds(std::max(start_error_ns, end_error_ns)),
            .frequency = frequency};
    } else {
        // Record spans more than one probe gap.
        const std::optional<uint64_t> end_index = chord_index_around(end_device_timestamp);
        TT_ASSERT(end_index.has_value());
        const Chord& start_chord = chord_at(*active_chord_index_);
        const Chord& end_chord = chord_at(*end_index);
        if (start_chord.rate_lo == 0.0 || end_chord.rate_lo == 0.0) {
            num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
        }
        const double start_host_ns = host_ns_on(start_chord, start_device_timestamp);
        const double end_host_ns = host_ns_on(end_chord, end_device_timestamp);
        const double secant =
            static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
        const int64_t placement_error_ns =
            std::max(error_ns_on(start_chord, start_device_timestamp), error_ns_on(end_chord, end_device_timestamp));
        // Publish the smoothed frequency with the skew priced, like the in-chord path — unless
        // the skew rivals the placement error, which only happens when a transition sits inside
        // the span and the record's own secant is the exact mapping.
        const double smoothed = end_chord.smoothed_frequency;
        const int64_t skew_ns = smoothed > 0.0
                                    ? static_cast<int64_t>(
                                          static_cast<double>(end_device_timestamp - start_device_timestamp) *
                                          std::abs(1.0 / smoothed - 1.0 / secant)) +
                                          1
                                    : std::numeric_limits<int64_t>::max();
        const bool use_smoothed = skew_ns <= placement_error_ns;
        const double frequency = use_smoothed ? smoothed : secant;
        mapping = RecordMapping{
            .device_cycle_offset =
                std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
            .error = std::chrono::nanoseconds(use_smoothed ? placement_error_ns + skew_ns : placement_error_ns),
            .frequency = frequency};
    }
    return mapping;
}

DeviceClockSync::DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core) :
    context_id_(context_id),
    chip_id_(device->id()),
    clock_core_virtual_(device->virtual_core_from_logical_core(clock_core, CoreType::WORKER)) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    configure_clock_read_path();
    if (mapped_clock_lo_ != nullptr) {
        // Throwaway cold read; the receiver warms the mapping up right before probing starts, so
        // no gap opens between warm-up and the steady cadence while other devices initialize.
        (void)probe();
    }
}

DeviceClockSync::~DeviceClockSync() = default;

void DeviceClockSync::configure_clock_read_path() {
    try {
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(chip_id_)->get_tlb_manager();
        if (tlb_manager == nullptr) {
            log_warning(
                tt::LogMetal,
                "[DeviceClockSync] Device {}: no TLB manager, so the clock register cannot be mapped",
                chip_id_);
            return;
        }
        tt::umd::tlb_data cfg{};
        cfg.local_offset = wall_clock_addr_lo_;
        cfg.x_end = clock_core_virtual_.x;
        cfg.y_end = clock_core_virtual_.y;
        cfg.ordering = tt::umd::tlb_data::Strict;
        clock_tlb_ = tlb_manager->allocate_tlb_window(cfg, tt::umd::TlbMapping::UC);
        if (clock_tlb_ == nullptr) {
            log_warning(
                tt::LogMetal,
                "[DeviceClockSync] Device {}: no UC TLB window available for the clock register",
                chip_id_);
            return;
        }
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal, "[DeviceClockSync] Device {}: could not map the clock register ({})", chip_id_, e.what());
    }
}

DeviceClockSync::Anchor DeviceClockSync::probe() {
    TTZoneScopedDN(RT_PROFILER, "Probe");
    const bool must_read_hi = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                              std::chrono::steady_clock::now() - last_probe_at_ > kMaxProbeGapBeforeRereadingHi;

    std::chrono::steady_clock::time_point host_before;
    std::chrono::steady_clock::time_point host_after;
    uint32_t lo = 0;

    {  // latency-critical
        host_before = std::chrono::steady_clock::now();
        tt_driver_atomics::lfence();
        lo = *mapped_clock_lo_;
        tt_driver_atomics::lfence();
        host_after = std::chrono::steady_clock::now();
    }

    if (must_read_hi) {
        cached_clock_hi_ = *mapped_clock_hi_;
    } else if (lo < last_clock_lo_) {
        ++cached_clock_hi_;
    }
    last_clock_lo_ = lo;
    last_probe_at_ = host_after;
    const auto bracket = host_after - host_before;
    TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(bracket.count()));
    const auto bracket_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bracket);
    const auto error = bracket_ns / 2;
    return Anchor{
        .host_timestamp = host_before + error,
        .device_timestamp = (static_cast<uint64_t>(cached_clock_hi_) << 32) | lo,
        .error = error};
}

DeviceClockSync::Anchor DeviceClockSync::read_probe() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    if (last_probe_at_ != std::chrono::steady_clock::time_point{}) {
        const int64_t gap_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - last_probe_at_)
                .count();
        if (gap_ns > peak_probe_gap_ns_.load(std::memory_order_relaxed)) {
            peak_probe_gap_ns_.store(gap_ns, std::memory_order_relaxed);
        }
    }
    // Best-of-N censors brackets an interrupt landed inside (host noise: a no-MMIO control
    // reproduces the same >100 us outliers); the EMA break self-quiets it when every read is slow.
    Anchor best = probe();
    for (int i = 1; i < kResyncProbes; i++) {
        if (typical_error_ > std::chrono::nanoseconds::zero() && best.error <= typical_error_ + typical_error_ / 2) {
            break;
        }
        const Anchor p = probe();
        if (p.error < best.error) {
            best = p;
        }
    }
    typical_error_ =
        typical_error_ == std::chrono::nanoseconds::zero() ? best.error : (typical_error_ * 7 + best.error) / 8;
    return best;
}

}  // namespace tt::tt_metal
