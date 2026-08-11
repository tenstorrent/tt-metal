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
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/device.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/tt_device/tt_device.hpp>
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

ClockSyncMapping::ClockSyncMapping(std::optional<FrequencyPrior> frequency_prior) : frequency_prior_(frequency_prior) {
    if (frequency_prior_.has_value() && !(frequency_prior_->min_frequency > 0.0 &&
                                          frequency_prior_->max_frequency >= frequency_prior_->min_frequency)) {
        frequency_prior_.reset();
    }
}

void ClockSyncMapping::set_fallback_step_bound(Chord& chord, double span_ns) const {
    // The observed envelope rather than the FrequencyPrior — the assumption bought and the trade
    // behind it are in the class contract.
    double lo = observed_min_frequency_;
    double hi = observed_max_frequency_;
    if (lo <= 0.0 && frequency_prior_.has_value()) {
        lo = frequency_prior_->min_frequency;
        hi = frequency_prior_->max_frequency;
    }
    if (lo <= 0.0) {
        // No rate knowledge at all: containment of both true and interpolated time in the chord,
        // with no distance refinement (threshold zero).
        chord.nonlinearity_ns = std::llround(span_ns);
        chord.refine_slope = 0.0;
        chord.refine_threshold_cycles = 0;
        return;
    }
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
    Chord& chord = chords_[slot];
    chord = Chord{
        .probe_error_ns = std::max(open.error, probe.error).count(),
        .frequency = frequency,
        .smoothed_frequency = frequency,
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

    if (observed_max_frequency_ == 0.0) {
        observed_min_frequency_ = rate_lo;
        observed_max_frequency_ = rate_hi;
    } else {
        observed_min_frequency_ = std::min(observed_min_frequency_, rate_lo);
        observed_max_frequency_ = std::max(observed_max_frequency_, rate_hi);
    }

    // Certificate: both two-chord windows around this chord, widened by read noise, must be
    // shorter than the minimum transition spacing. Then at most one transition touches this
    // chord, its neighbors are transition-free, and the brackets above contain every rate inside.
    const auto back_window = (close.host_timestamp - outer_open.host_timestamp) + outer_open.error + close.error;
    const auto forward_window = (outer_close.host_timestamp - open.host_timestamp) + open.error + outer_close.error;
    if (back_window >= kDvfsMinTransitionSpacing || forward_window >= kDvfsMinTransitionSpacing) {
        frequency_window_active_ = false;
        // Holdback publishes records only after finalize, so this restamp — not the creation-time
        // one — is the bound they see; the fold above just widened the envelope with this chord's
        // own neighbor secants.
        set_fallback_step_bound(chord, span_ns);
        return;
    }

    chord.nonlinearity_ns = transition_step_bound_ns(span_ns, rate_hi / rate_lo);
    chord.refine_slope = 1.0 / rate_lo - 1.0 / rate_hi;
    chord.refine_threshold_cycles = refine_threshold_for(chord.nonlinearity_ns, chord.refine_slope);
    chord.rate_lo = rate_lo;
    chord.rate_hi = rate_hi;
    num_certified_chords_.fetch_add(1, std::memory_order_relaxed);

    // Frequency window: extend while this chord's bracket intersects the running intersection —
    // then every rate since the anchor lies inside it and the window secant is a faithful average.
    // An empty intersection is a detected transition; the window restarts at this chord.
    if (frequency_window_active_ && frequency_window_anchor_ >= begin &&
        std::max(frequency_window_rate_lo_, rate_lo) <= std::min(frequency_window_rate_hi_, rate_hi)) {
        frequency_window_rate_lo_ = std::max(frequency_window_rate_lo_, rate_lo);
        frequency_window_rate_hi_ = std::min(frequency_window_rate_hi_, rate_hi);
        slide_frequency_window(close_index);
    } else {
        frequency_window_active_ = true;
        frequency_window_anchor_ = close_index - 1;
        frequency_window_rate_lo_ = rate_lo;
        frequency_window_rate_hi_ = rate_hi;
    }
    const Anchor& anchor = probe_at(frequency_window_anchor_);
    chord.smoothed_frequency =
        static_cast<double>(close.device_timestamp - anchor.device_timestamp) / span_ns_between(anchor, close);
    chord.smoothing_skew_per_cycle = std::abs(1.0 / chord.smoothed_frequency - 1.0 / chord.frequency);
}

void ClockSyncMapping::slide_frequency_window(uint64_t close_index) {
    const Anchor& close = probe_at(close_index);
    if (close.host_timestamp - probe_at(frequency_window_anchor_).host_timestamp <= kFrequencyWindowMax) {
        return;
    }
    const auto slide_deadline =
        close.host_timestamp - std::chrono::duration_cast<std::chrono::steady_clock::duration>(kFrequencyWindowSlide);
    uint64_t new_anchor = frequency_window_anchor_;
    while (new_anchor + 1 < close_index && probe_at(new_anchor + 1).host_timestamp <= slide_deadline) {
        ++new_anchor;
    }
    double rate_lo = 0.0;
    double rate_hi = std::numeric_limits<double>::infinity();
    for (uint64_t j = new_anchor + 1; j <= close_index; ++j) {
        rate_lo = std::max(rate_lo, chord_at(j).rate_lo);
        rate_hi = std::min(rate_hi, chord_at(j).rate_hi);
    }
    frequency_window_anchor_ = new_anchor;
    frequency_window_rate_lo_ = rate_lo;
    frequency_window_rate_hi_ = rate_hi;
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

bool ClockSyncMapping::pin_start(uint64_t device_timestamp) {
    if (device_timestamp == 0) {
        return false;
    }
    if (device_timestamp == last_pin_device_timestamp_) {
        return true;
    }
    const std::optional<uint64_t> close_index = chord_index_around(device_timestamp);
    if (!close_index.has_value() || *close_index + 1 >= probes_end_) {
        // No chord yet, or the chord is not finalized; the caller re-offers within a probe interval.
        return false;
    }
    last_pin_device_timestamp_ = device_timestamp;
    const Chord& chord = chord_at(*close_index);
    pinned_start_ = Anchor{
        .host_timestamp = std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(std::llround(host_ns_on(chord, device_timestamp)))),
        .device_timestamp = device_timestamp,
        .error = std::chrono::nanoseconds(error_ns_on(chord, device_timestamp)),
    };
    return true;
}

std::optional<ClockSyncMapping::RecordMapping> ClockSyncMapping::map_record(
    uint64_t start_device_timestamp, uint64_t end_device_timestamp) {
    const uint64_t begin = oldest_probe();
    const bool active_chord_valid = active_chord_index_.has_value() && *active_chord_index_ > begin &&
                                    *active_chord_index_ < probes_end_ &&
                                    start_device_timestamp > chord_at(*active_chord_index_).open_device_timestamp &&
                                    start_device_timestamp <= chord_at(*active_chord_index_).close_device_timestamp;
    if (!active_chord_valid) {
        active_chord_index_ = chord_index_around(start_device_timestamp);
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
        if (end_chord.rate_lo == 0.0) {
            num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
        }
        if (pinned_start_.has_value() && pinned_start_->device_timestamp == start_device_timestamp) {
            // Long program: reuse the host time we pinned while it was still running.
            const double start_host_ns = static_cast<double>(pinned_start_->host_timestamp.time_since_epoch().count());
            const double frequency =
                static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
            mapping = RecordMapping{
                .device_cycle_offset =
                    std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
                .error = std::chrono::nanoseconds(std::max(pinned_start_->error.count(), end_error_ns)),
                .frequency = frequency};
        } else {
            // No pinned start: ride the cycle count back from the oldest retained probe at the
            // envelope of every rate this clock has been observed at. The pre-history rate is
            // unmeasured — the error assumes it stayed inside that envelope.
            num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
            double env_lo = observed_min_frequency_;
            double env_hi = observed_max_frequency_;
            if (!(env_lo > 0.0)) {
                env_lo = frequency_prior_.has_value() ? frequency_prior_->min_frequency : end_chord.frequency;
                env_hi = frequency_prior_.has_value() ? frequency_prior_->max_frequency : end_chord.frequency;
            }
            const Anchor& ring_oldest = probe_at(begin);
            const double uncovered_cycles = static_cast<double>(ring_oldest.device_timestamp - start_device_timestamp);
            const double start_host_ns = host_ns(ring_oldest) - uncovered_cycles * (1.0 / env_lo + 1.0 / env_hi) * 0.5;
            const int64_t start_error_ns =
                std::llround(uncovered_cycles * (1.0 / env_lo - 1.0 / env_hi) * 0.5) + ring_oldest.error.count();
            const double frequency =
                static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
            mapping = RecordMapping{
                .device_cycle_offset =
                    std::llround(static_cast<double>(end_device_timestamp) - frequency * end_host_ns),
                .error = std::chrono::nanoseconds(std::max(start_error_ns, end_error_ns)),
                .frequency = frequency};
        }
    } else if (const Chord& chord = chord_at(*active_chord_index_);
               end_device_timestamp <= chord.close_device_timestamp) {
        // Usual case: whole record fits between two adjacent probes. The published frequency is the
        // window-smoothed one; consumers reconstruct host_end from it, so the (exactly known) skew
        // against the interpolation slope goes into the error.
        const double start_host_ns = host_ns_on(chord, start_device_timestamp);
        if (chord.rate_lo == 0.0) {
            num_records_on_uncertified_chords_.fetch_add(1, std::memory_order_relaxed);
        }
        int64_t end_error_ns = error_ns_on(chord, end_device_timestamp);
        if (chord.smoothing_skew_per_cycle != 0.0) {
            end_error_ns += static_cast<int64_t>(
                                static_cast<double>(end_device_timestamp - start_device_timestamp) *
                                chord.smoothing_skew_per_cycle) +
                            1;
        }
        mapping = RecordMapping{
            .device_cycle_offset =
                std::llround(static_cast<double>(start_device_timestamp) - chord.smoothed_frequency * start_host_ns),
            .error = std::chrono::nanoseconds(std::max(error_ns_on(chord, start_device_timestamp), end_error_ns)),
            .frequency = chord.smoothed_frequency};
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
        const double frequency =
            static_cast<double>(end_device_timestamp - start_device_timestamp) / (end_host_ns - start_host_ns);
        mapping = RecordMapping{
            .device_cycle_offset =
                std::llround(static_cast<double>(start_device_timestamp) - frequency * start_host_ns),
            .error = std::chrono::nanoseconds(std::max(
                error_ns_on(start_chord, start_device_timestamp), error_ns_on(end_chord, end_device_timestamp))),
            .frequency = frequency};
    }
    if (pinned_start_.has_value() && start_device_timestamp >= pinned_start_->device_timestamp) {
        pinned_start_.reset();
    }
    return mapping;
}

namespace {

std::optional<ClockSyncMapping::FrequencyPrior> fetch_frequency_prior(ContextId context_id, uint32_t chip_id) {
    try {
        auto* tt_device = MetalContext::instance(context_id).get_cluster().get_driver()->get_tt_device(chip_id);
        if (tt_device == nullptr) {
            return std::nullopt;
        }
        const double min_frequency = static_cast<double>(tt_device->get_min_clock_freq()) / 1000.0;  // MHz -> GHz
        const double max_frequency = static_cast<double>(tt_device->get_max_clock_freq()) / 1000.0;
        if (!(min_frequency > 0.0) || max_frequency < min_frequency) {
            return std::nullopt;
        }
        return ClockSyncMapping::FrequencyPrior{.min_frequency = min_frequency, .max_frequency = max_frequency};
    } catch (const std::exception& e) {
        log_debug(tt::LogMetal, "[DeviceClockSync] Device {}: no AICLK range available ({})", chip_id, e.what());
        return std::nullopt;
    }
}

}  // namespace

DeviceClockSync::DeviceClockSync(ContextId context_id, IDevice* device, CoreCoord clock_core) :
    context_id_(context_id),
    chip_id_(device->id()),
    clock_core_virtual_(device->virtual_core_from_logical_core(clock_core, CoreType::WORKER)),
    mapping_(fetch_frequency_prior(context_id, device->id())) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    configure_clock_read_path();
    if (mapped_clock_lo_ != nullptr) {
        // Throwaway cold read, then spaced probes so map_record already has finalized chords.
        constexpr int kWarmUpProbes = 4;
        (void)probe();
        for (int i = 0; i < kWarmUpProbes; i++) {
            if (i != 0) {
                std::this_thread::sleep_for(kDeviceClockSyncInterval);
            }
            resync();
        }
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

void DeviceClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    if (last_probe_at_ != std::chrono::steady_clock::time_point{}) {
        const int64_t gap_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - last_probe_at_)
                .count();
        if (gap_ns > peak_probe_gap_ns_.load(std::memory_order_relaxed)) {
            peak_probe_gap_ns_.store(gap_ns, std::memory_order_relaxed);
        }
    }
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
    mapping_.add_probe(best);
}

}  // namespace tt::tt_metal
