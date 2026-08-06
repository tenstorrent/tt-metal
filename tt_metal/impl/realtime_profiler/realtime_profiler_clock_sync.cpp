// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include <tt-metalium/device.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/tlb.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/common/env_lib.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

// ~0.5s at these values.
constexpr uint32_t kFitProbes = 100;
constexpr auto kProbeInterval = std::chrono::milliseconds(5);
constexpr auto kSettleDelay = std::chrono::milliseconds(50);
constexpr int kProbesPerPoint = 4;

constexpr auto kCalibrationCacheMaxAge = std::chrono::seconds(60);

// The counter could have been read anywhere inside the bracket.
constexpr std::chrono::nanoseconds placement_error(std::chrono::nanoseconds bracket) { return bracket / 2; }

// Loose enough that ordinary spread survives; only reads serviced late are cut.
constexpr int kFitBracketOutlierFactor = 2;

// Lets a rapid MeshDevice reopen skip the bring-up fit: the device WALL_CLOCK free-runs across close, so the rate
// measured last time still holds.
struct FrequencyCache {
    struct Entry {
        double frequency = 0.0;
        std::chrono::steady_clock::time_point updated_at;
    };
    std::mutex mutex;
    std::unordered_map<uint32_t, Entry> by_chip;
};

FrequencyCache& frequency_cache() {
    static ttsl::Indestructible<FrequencyCache> cache;
    return cache.get();
}

}  // namespace

std::chrono::nanoseconds RealtimeProfilerClockSync::sync_interval() {
    static const std::chrono::nanoseconds interval =
        std::chrono::microseconds(tt::parse_env<uint32_t>("TT_RT_PROFILER_SYNC_INTERVAL_US", 500));
    return interval;
}

void RealtimeProfilerClockModel::seed_frequency(double frequency) {
    TT_FATAL(frequency > 0.0, "Real-time profiler clock model needs a positive seed frequency, got {}", frequency);
    frequency_ = frequency;
}

void RealtimeProfilerClockModel::adopt_rate(double rate) {
    if (rate <= 0.0) {
        return;
    }
    frequency_ = rate;
}

std::optional<RealtimeProfilerClockModel::FitResidual> RealtimeProfilerClockModel::fit(
    std::span<const ClockProbe> probes, std::chrono::steady_clock::time_point host_start) {
    if (probes.size() < 2) {
        return std::nullopt;
    }

    std::vector<std::chrono::nanoseconds> brackets;
    brackets.reserve(probes.size());
    for (const auto& p : probes) {
        brackets.push_back(p.bracket);
    }
    const auto median = brackets.begin() + static_cast<ptrdiff_t>(brackets.size() / 2);
    std::nth_element(brackets.begin(), median, brackets.end());
    const std::chrono::nanoseconds median_bracket = *median;

    std::vector<ClockProbe> tight;
    tight.reserve(probes.size());
    for (const auto& p : probes) {
        if (p.bracket <= median_bracket * kFitBracketOutlierFactor) {
            tight.push_back(p);
        }
    }
    const std::span<const ClockProbe> fitted_probes = tight.size() >= 2 ? std::span<const ClockProbe>(tight) : probes;

    // Centered on host_start to avoid catastrophic cancellation from regressing at absolute-timestamp magnitudes.
    const double n = static_cast<double>(fitted_probes.size());
    const auto centered_host = [host_start](const ClockProbe& p) {
        return static_cast<double>((p.host_time + placement_error(p.bracket) - host_start).count());
    };

    double host_mean = 0.0;
    double device_mean = 0.0;
    for (const auto& p : fitted_probes) {
        host_mean += centered_host(p);
        device_mean += static_cast<double>(p.device_ticks);
    }
    host_mean /= n;
    device_mean /= n;

    double num = 0.0;
    double den = 0.0;
    for (const auto& p : fitted_probes) {
        const double dx = centered_host(p) - host_mean;
        const double dy = static_cast<double>(p.device_ticks) - device_mean;
        num += dx * dy;
        den += dx * dx;
    }
    if (std::abs(den) > 1e-10) {
        const double fitted = num / den;
        // Consumers divide by this, so an unusable slope must not reach them; the seeded AICLK still maps.
        if (fitted > 0.0) {
            frequency_ = fitted;
        } else {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Clock fit produced a non-positive frequency ({}); keeping the commanded AICLK",
                fitted);
        }
    }

    FitResidual residual;
    double residual_sumsq_ns = 0.0;
    for (const auto& p : fitted_probes) {
        const double predicted = device_mean + frequency_ * (centered_host(p) - host_mean);
        const double residual_ns = (static_cast<double>(p.device_ticks) - predicted) / frequency_;
        residual_sumsq_ns += residual_ns * residual_ns;
        residual.max_ns = std::max(residual.max_ns, std::abs(residual_ns));
    }
    residual.rms_ns = std::sqrt(residual_sumsq_ns / n);
    residual.num_probes_fitted = fitted_probes.size();
    residual.num_probes_offered = probes.size();

    return residual;
}

RealtimeProfilerClockSync::RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core) :
    context_id_(context_id),
    chip_id_(device->id()),
    profiler_core_virtual_(device->virtual_core_from_logical_core(profiler_core, CoreType::WORKER)) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    model_.seed_frequency(MetalContext::instance(context_id_).get_cluster().get_device_aiclk(chip_id_) / 1000.0);
    configure_clock_read_path();
}

RealtimeProfilerClockSync::~RealtimeProfilerClockSync() = default;

void RealtimeProfilerClockSync::configure_clock_read_path() {
    try {
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(chip_id_)->get_tlb_manager();
        if (tlb_manager == nullptr) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: no TLB manager, so the clock register cannot be mapped",
                chip_id_);
            return;
        }
        tt::umd::tlb_data cfg{};
        cfg.local_offset = wall_clock_addr_lo_;
        cfg.x_end = profiler_core_virtual_.x;
        cfg.y_end = profiler_core_virtual_.y;
        cfg.ordering = tt::umd::tlb_data::Strict;
        clock_tlb_ = tlb_manager->allocate_tlb_window(cfg, tt::umd::TlbMapping::UC);
        if (clock_tlb_ == nullptr) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: no UC TLB window available for the clock register",
                chip_id_);
            return;
        }
        // Resolved once for sync-latency purposes
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal, "[Real-time profiler] Device {}: could not map the clock register ({})", chip_id_, e.what());
    }
}

std::optional<RealtimeProfilerClockSync::BaselineRate> RealtimeProfilerClockSync::baseline_rate() const {
    const uint64_t begin = oldest_probe();
    if (probes_end_ - begin < 2) {
        return std::nullopt;
    }
    const Anchor& newest = probe_at(probes_end_ - 1);
    // Walk back to the newest probe that is still at least kRateBaseline older than the newest one. That, not the
    // ring's oldest entry, is the near end of the baseline: how far back the rate is measured has to be a property of
    // the rate, not of how much history the ring happens to be holding.
    const auto cutoff = newest.host - kRateBaseline;
    uint64_t near = probes_end_ - 1;
    while (near > begin && probe_at(near).host > cutoff) {
        --near;
    }
    const Anchor& oldest = probe_at(near);
    if (newest.ticks <= oldest.ticks || newest.host <= oldest.host) {
        return std::nullopt;
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(newest.host - oldest.host).count());
    // A baseline this narrow is no tighter than the chord it exists to improve on, so the chord's own slope stands.
    if (newest.host - oldest.host < kRateBaseline / 4) {
        return std::nullopt;
    }
    return BaselineRate{
        .rate = static_cast<double>(newest.ticks - oldest.ticks) / span_ns,
        .noise = static_cast<double>((oldest.bracket + newest.bracket).count()) / span_ns,
    };
}

uint64_t RealtimeProfilerClockSync::first_probe_at_or_past(uint64_t ticks) const {
    uint64_t lo = oldest_probe();
    uint64_t hi = probes_end_;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (probe_at(mid).ticks < ticks) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::place(uint64_t ticks) {
    if (probes_end_ == 0) {
        return std::nullopt;
    }
    const uint64_t probes_begin = oldest_probe();
    const uint64_t close_index = first_probe_at_or_past(ticks);
    const auto baseline = baseline_rate();

    // Past every retained probe. The caller probes after reading a batch, so this is not the ordinary case for a record
    // -- it means the read raced the probe, or the probe failed -- and there is nothing to wait for either way, since
    // this is asked once, on the pass that read the record.
    if (close_index == probes_end_) {
        return extrapolate_from(probe_at(probes_end_ - 1), baseline);
    }

    const Anchor& closing = probe_at(close_index);

    // Near side: the newest probe far enough from the far side to take a slope from. Walking back is monotone in span
    // and the span floor is the only thing plan_chord_mapping refuses on, so the first anchor that clears it is both
    // the tightest usable pair and the last word -- an older one cannot succeed where this failed.
    for (uint64_t i = close_index; i-- > probes_begin;) {
        if (closing.host - probe_at(i).host < sync_interval() / 2) {
            continue;
        }
        if (auto chord = plan_chord_mapping(probe_at(i), closing, baseline, previous_rate_, previous_rate_noise_)) {
            if (closing.ticks != current_chord_close_ticks_) {
                previous_rate_ = current_rate_;
                previous_rate_noise_ = current_rate_noise_;
                current_chord_close_ticks_ = closing.ticks;
            }
            current_rate_ = chord->chord_rate;
            current_rate_noise_ = chord->chord_rate_noise;
            last_published_sync_error_ = chord->mapping.sync_error;
            model_.adopt_rate(chord->frequency);
            return chord;
        }
        break;
    }

    return extrapolate_from(closing, baseline);
}

// Deliberately does not touch previous_rate_: no slope was measured here, and feeding an unmeasured one into the next
// interval's curvature term would invent a DVFS step.
std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::extrapolate_from(
    const Anchor& anchor, const std::optional<BaselineRate>& baseline) {
    const double rate = baseline.has_value() && baseline->rate > 0.0 ? baseline->rate : frequency();
    // Without a baseline the rate is the bring-up fit's, whose residual is quoted in ppm; a whole percent is a
    // deliberate over-estimate, and this runs only when the history cannot produce a pair at all.
    const double rate_noise = baseline.has_value() ? baseline->noise : 0.01;
    if (rate <= 0.0) {
        return std::nullopt;
    }
    ChordMapping anchored{
        .mapping = {.device_cycle_offset = 0, .sync_error = placement_error(anchor.bracket)},
        .frequency = rate,
        .chord_rate = rate,
        .chord_rate_noise = rate_noise,
        .open_ticks = anchor.ticks,
        .open_host_ns = static_cast<double>(anchor.host.time_since_epoch().count()),
        .inv_chord_rate = 1.0 / rate,
        .close_ticks = anchor.ticks,
        // Zero span, so every timestamp is outside it and place_on_chord charges each one its own distance from the
        // anchor. That is what makes a single anchor safe to reuse across a whole backlog.
        .batch_through_ticks = std::numeric_limits<uint64_t>::max(),
    };
    last_published_sync_error_ = anchored.mapping.sync_error;
    return anchored;
}

std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::plan_chord_mapping(
    const Anchor& open,
    const Anchor& closing,
    const std::optional<BaselineRate>& baseline,
    double previous_rate,
    double previous_rate_noise) {
    // A chord much shorter than the interval asked for places records inside it just fine, but its slope is uncertain
    // by (both brackets)/span and that slope is published as `frequency`. A consumer evaluating the mapping away from
    // the record would then see slope_noise * distance, which is how a few-us chord becomes microseconds of error.
    //
    // Non-monotone ticks are refused with it, which is the whole of what a pair is checked for: the counter is read low
    // word first and that latches the high word (see the arch c_tensix_core.h), so the composed 64-bit value cannot
    // tear on either side, and a probe landing somewhere the clock could not have been is not a thing that happens.
    if (closing.host <= open.host || closing.ticks <= open.ticks || closing.host - open.host < sync_interval() / 2) {
        return std::nullopt;
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(closing.host - open.host).count());
    const double rate = static_cast<double>(closing.ticks - open.ticks) / span_ns;
    const double rate_noise = static_cast<double>((open.bracket + closing.bracket).count()) / span_ns;

    // A step of relative size D at fraction L of the interval puts the true trajectory D*T*L*(1-L) off the chord, up
    // to D*T/4. D is not observable; how much this interval's rate differs from the last one's is, and for a step at
    // the midpoint that difference is D/2, hence T/2. Only the part neither measurement could have invented counts, so
    // both secants' noise comes off first -- a short interval's slope is uncertain enough to fake a whole DVFS step on
    // its own. Reads zero on a plateau, which is nearly always.
    std::chrono::nanoseconds curvature{};
    if (previous_rate > 0.0) {
        const double relative_rate_change = std::abs(rate - previous_rate) / rate;
        const double attributable = std::max(0.0, relative_rate_change - rate_noise - previous_rate_noise);
        curvature = std::chrono::nanoseconds(static_cast<int64_t>(span_ns * attributable / 2.0));
    }

    // The rate published to consumers is the baseline's, not this chord's: a chord this narrow measures its slope to
    // only a few thousand ppm, and every duration a consumer computes divides by it. Placement is unaffected because
    // each record is anchored to where this chord puts it -- see place_on_chord.
    const double published_rate = baseline.has_value() && baseline->rate > 0.0 ? baseline->rate : rate;

    return ChordMapping{
        .mapping =
            experimental::ProgramRealtimeClockSync{
                .device_cycle_offset = 0,  // per record
                .sync_error = interpolation_error(open, closing) + curvature,
            },
        .frequency = published_rate,
        .chord_rate = rate,
        .chord_rate_noise = rate_noise,
        .open_ticks = open.ticks,
        .open_host_ns = static_cast<double>(open.host.time_since_epoch().count()),
        .inv_chord_rate = 1.0 / rate,
        .close_ticks = closing.ticks,
        .batch_through_ticks = closing.ticks,
    };
}

std::optional<ClockProbe> RealtimeProfilerClockSync::probe() {
    TTZoneScopedDN(RT_PROFILER, "Probe");
    uint32_t lo = 0;
    try {
        if (model_.frequency() != wrap_period_frequency_) {
            wrap_period_frequency_ = model_.frequency();
            wrap_period_ =
                std::chrono::nanoseconds(static_cast<int64_t>((1ull << 32) / std::max(wrap_period_frequency_, 0.1)));
        }
        // Halved for a safety margin.
        const bool wrap_could_have_been_missed = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                                                 std::chrono::steady_clock::now() - last_probe_at_ > wrap_period_ / 2;

        std::chrono::steady_clock::time_point host_before;
        std::chrono::steady_clock::time_point host_after;
        {  // latency-critical
            host_before = std::chrono::steady_clock::now();
            lo = *mapped_clock_lo_;
            host_after = std::chrono::steady_clock::now();
        }
        // The high word only moves when the low word wraps, and reading the low word latches it, so it stays outside
        // the bracket.
        if (wrap_could_have_been_missed || lo < last_clock_lo_) {
            cached_clock_hi_ = *mapped_clock_hi_;
        }
        last_clock_lo_ = lo;
        last_probe_at_ = host_after;
        ++cost_.clock_reads;
        const uint32_t hi = cached_clock_hi_;
        const auto bracket = host_after - host_before;
        TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(bracket.count()));
        return ClockProbe{
            host_before,
            std::chrono::duration_cast<std::chrono::nanoseconds>(bracket),
            (static_cast<uint64_t>(hi) << 32) | lo};
    } catch (const std::exception& e) {
        log_debug(tt::LogMetal, "[Real-time profiler] Device {}: clock read failed ({})", chip_id_, e.what());
        return std::nullopt;
    }
}

std::optional<ClockProbe> RealtimeProfilerClockSync::best_of(int probes) {
    std::optional<ClockProbe> best;
    for (int i = 0; i < probes; i++) {
        const auto p = probe();
        if (p.has_value() && (!best.has_value() || p->bracket < best->bracket)) {
            best = p;
        }
        // Each read blocks the calling thread on PCIe, so the remaining ones are only worth taking while they might
        // still tighten the bracket. A read already at the recent typical width leaves them nothing to improve; the
        // full count is spent only when the link is making reads late.
        if (best.has_value() && typical_bracket_ > std::chrono::nanoseconds::zero() &&
            best->bracket <= typical_bracket_ + typical_bracket_ / 2) {
            break;
        }
    }
    if (best.has_value()) {
        typical_bracket_ = typical_bracket_ == std::chrono::nanoseconds::zero()
                               ? best->bracket
                               : (typical_bracket_ * 7 + best->bracket) / 8;
    }
    return best;
}

void RealtimeProfilerClockSync::bring_up() {
    TTZoneScopedDN(RT_PROFILER, "ClockBringUp");
    constexpr uint32_t kMaxRetries = 3;
    constexpr auto kRetryDelay = std::chrono::milliseconds(500);

    if (try_cached_calibration()) {
        return;
    }
    for (uint32_t attempt = 0; attempt <= kMaxRetries; attempt++) {
        if (attempt > 0) {
            log_debug(tt::LogMetal, "[Real-time profiler] Device {} sync retry {}/{}", chip_id_, attempt, kMaxRetries);
            std::this_thread::sleep_for(kRetryDelay);
        }
        if (calibrate()) {
            return;
        }
    }
}

bool RealtimeProfilerClockSync::try_cached_calibration() {
    TTZoneScopedDN(RT_PROFILER, "RestoreCalibration");
    const auto now = std::chrono::steady_clock::now();
    std::optional<double> frequency;
    {
        auto& cache = frequency_cache();
        std::lock_guard lock(cache.mutex);
        const auto it = cache.by_chip.find(chip_id_);
        if (it != cache.by_chip.end() && now - it->second.updated_at < kCalibrationCacheMaxAge) {
            frequency = it->second.frequency;
        }
    }
    if (!frequency.has_value()) {
        return false;
    }

    model_.seed_frequency(*frequency);
    if (!resync()) {
        return false;  // the read failed, so fall back to a full fit
    }
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device {}: reusing cached clock frequency (fit within {}s), skipping the multi-probe fit",
        chip_id_,
        static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(kCalibrationCacheMaxAge).count()));
    return true;
}

bool RealtimeProfilerClockSync::calibrate() {
    TTZoneScopedDN(RT_PROFILER, "Calibrate");
    constexpr uint32_t kMaxConsecutiveFailures = 3;
    // A settled clock fits to ~2ns rms; a fit across an AICLK ramp still looks well-conditioned but lands tens of
    // ppm off, giving a residual ~1000x higher.
    constexpr double kMaxFitResidualRmsNs = 200.0;
    const auto host_start_time = std::chrono::steady_clock::now();

    std::vector<ClockProbe> probes;
    probes.reserve(kFitProbes);
    std::this_thread::sleep_for(kSettleDelay);

    // Warms the cold PCIe path first; otherwise it lands at one end of the span with outsized leverage over the
    // slope.
    (void)probe();

    uint32_t consecutive_failures = 0;
    for (uint32_t attempt = 0; attempt < kFitProbes; attempt++) {
        std::this_thread::sleep_for(kProbeInterval);
        const auto p = best_of(kProbesPerPoint);
        if (!p.has_value()) {
            if (++consecutive_failures >= kMaxConsecutiveFailures) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} sync aborted after {} consecutive failed clock reads",
                    chip_id_,
                    consecutive_failures);
                break;
            }
            continue;
        }
        consecutive_failures = 0;
        probes.push_back(*p);
    }

    const std::optional<RealtimeProfilerClockModel::FitResidual> residual = model_.fit(probes, host_start_time);
    // fit() always sets the anchor, even though the fit may be rejected below: records can drain before bring-up
    // finishes, so the mapping must stay readable regardless.
    if (!residual.has_value()) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough probes, using the commanded AICLK",
            chip_id_);
        return false;
    }

    // Checked against kFitProbes, not what was collected, to catch both an unreadable link and probes the model
    // discarded.
    if (residual->num_probes_fitted * 2 < kFitProbes) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} fit only {} of {} wanted sync probes; retrying rather than fitting a "
            "frequency from what is left",
            chip_id_,
            residual->num_probes_fitted,
            kFitProbes);
        return false;
    }

    if (residual->rms_ns > kMaxFitResidualRmsNs) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} clock fit residual is {:.0f} ns rms, past the {:.0f} ns a settled clock "
            "gives; the frequency was likely fitted across an AICLK change, so retrying",
            chip_id_,
            residual->rms_ns,
            kMaxFitResidualRmsNs);
        return false;
    }

    // Only once the fit is worth reusing: a bad one would outlive the run that produced it.
    {
        auto& cache = frequency_cache();
        std::lock_guard lock(cache.mutex);
        cache.by_chip[chip_id_] = FrequencyCache::Entry{model_.frequency(), std::chrono::steady_clock::now()};
    }
    log_info(
        tt::LogMetal,
        "[Real-time profiler] Device {} sync complete: fit {} of {} collected probes, frequency={:.6f} GHz, fit "
        "residual rms={:.0f} ns max={:.0f} ns",
        chip_id_,
        residual->num_probes_fitted,
        residual->num_probes_offered,
        model_.frequency(),
        residual->rms_ns,
        residual->max_ns);
    return true;
}

bool RealtimeProfilerClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    const auto started_at = std::chrono::steady_clock::now();
    const auto p = best_of(kResyncProbes);
    ++cost_.resyncs;
    if (p.has_value()) {
        probe_history_[probes_end_ % kProbeHistoryCapacity] =
            Anchor{p->host_time + placement_error(p->bracket), p->device_ticks, p->bracket};
        ++probes_end_;
    }
    cost_.busy += std::chrono::steady_clock::now() - started_at;
    return p.has_value();
}

}  // namespace tt::tt_metal
