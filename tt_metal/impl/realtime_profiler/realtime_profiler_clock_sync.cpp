// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
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

void RealtimeProfilerClockModel::seed_frequency(double frequency) {
    TT_FATAL(frequency > 0.0, "Real-time profiler clock model needs a positive seed frequency, got {}", frequency);
    frequency_ = frequency;
    seed_frequency_ = frequency;
}

void RealtimeProfilerClockModel::adopt_rate(double rate) {
    if (rate <= 0.0 ||
        std::abs(rate - seed_frequency_) > seed_frequency_ * RealtimeProfilerClockSync::kRateClampFraction) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Measured clock rate {} outside the band around the commanded {}; keeping {}",
            rate,
            seed_frequency_,
            frequency_);
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
            return;
        }
        tt::umd::tlb_data cfg{};
        cfg.local_offset = wall_clock_addr_lo_;
        cfg.x_end = profiler_core_virtual_.x;
        cfg.y_end = profiler_core_virtual_.y;
        cfg.ordering = tt::umd::tlb_data::Strict;
        clock_tlb_ = tlb_manager->allocate_tlb_window(cfg, tt::umd::TlbMapping::UC);
        if (clock_tlb_ == nullptr) {
            return;
        }
        // Resolved once for sync-latency purposes
        const uint64_t local = clock_tlb_->handle_ref().get_config().local_offset;
        auto* base = clock_tlb_->handle_ref().get_base();
        mapped_clock_lo_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_lo_ - local));
        mapped_clock_hi_ = reinterpret_cast<volatile uint32_t*>(base + (wall_clock_addr_hi_ - local));
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: no TLB window for the clock register ({}); sync reads it through UMD",
            chip_id_,
            e.what());
    }
}

void RealtimeProfilerClockSync::retire_probes_before(uint64_t ticks) {
    // Two probes are kept preceding `ticks`, not one: a near side has to be far enough from the far side to take a
    // slope from, and a single retained probe cannot span anything.
    while (probes_.size() > 3 && probes_[2].ticks <= ticks) {
        probes_.pop_front();
    }
}

std::optional<std::pair<RealtimeProfilerClockSync::Anchor, RealtimeProfilerClockSync::Anchor>>
RealtimeProfilerClockSync::probes_bracketing(uint64_t start_ticks, uint64_t end_ticks) const {
    if (probes_.size() < 2) {
        return std::nullopt;
    }
    // Probes are recorded in order, so both ends are found by bisection: the retained span grows with the backlog, and
    // scanning it per record is what turns a backlog into a stall.
    const auto by_ticks = [](const Anchor& a, uint64_t ticks) { return a.ticks < ticks; };

    // Far side: the oldest probe that still reads past the record, which keeps the chord as short as it can be.
    const auto close_it = std::lower_bound(probes_.begin(), probes_.end(), end_ticks, by_ticks);
    if (close_it == probes_.end()) {
        // No probe has read past this record yet. It waits, which is the one thing worth waiting for.
        return std::nullopt;
    }
    size_t close_index = static_cast<size_t>(close_it - probes_.begin());
    if (close_index == 0) {
        // The record predates everything retained; the two oldest are the closest thing to a chord around it.
        close_index = 1;
    }

    // Near side: the newest probe at or before the record that is also far enough from the far side to take a slope
    // from. The span requirement is applied here so a pair is never offered that plan_chord_mapping would refuse.
    const Anchor& close_anchor = probes_[close_index];
    const auto start_it = std::upper_bound(
        probes_.begin(), probes_.begin() + close_index, start_ticks, [](uint64_t ticks, const Anchor& a) {
            return ticks < a.ticks;
        });
    size_t open_index = 0;
    for (size_t i = static_cast<size_t>(start_it - probes_.begin()); i-- > 0;) {
        if (close_anchor.host - probes_[i].host >= kSyncInterval / 2) {
            open_index = i;
            break;
        }
    }
    return std::make_pair(probes_[open_index], close_anchor);
}

std::optional<RealtimeProfilerClockSync::ChordMapping> RealtimeProfilerClockSync::plan_chord_mapping(
    const Anchor& open, const Anchor& closing, double previous_rate, double previous_rate_noise, double sanity_rate) {
    // A chord much shorter than the interval asked for places records inside it just fine, but its slope is uncertain
    // by (both brackets)/span and that slope is published as `frequency`. A consumer evaluating the mapping away from
    // the record would then see slope_noise * distance, which is how a few-us chord becomes microseconds of error.
    if (closing.host <= open.host || closing.ticks <= open.ticks || closing.host - open.host < kSyncInterval / 2) {
        return std::nullopt;
    }
    const double span_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(closing.host - open.host).count());
    const double rate = static_cast<double>(closing.ticks - open.ticks) / span_ns;
    // A chord this far from the last measured rate means one of its two probes is not where it claims.
    if (sanity_rate > 0.0 && std::abs(rate - sanity_rate) >= sanity_rate * kRateClampFraction) {
        return std::nullopt;
    }

    // A step of relative size D at fraction L of the interval puts the true trajectory D*T*L*(1-L) off the chord, up
    // to D*T/4. D is not observable; how much this interval's rate differs from the last one's is, and for a step at
    // the midpoint that difference is D/2, hence T/2. Only the part neither measurement could have invented counts, so
    // both secants' noise comes off first -- a short interval's slope is uncertain enough to fake a whole DVFS step on
    // its own. Reads zero on a plateau, which is nearly always.
    const double rate_noise = static_cast<double>((open.bracket + closing.bracket).count()) / span_ns;
    std::chrono::nanoseconds curvature{};
    if (previous_rate > 0.0) {
        const double relative_rate_change = std::abs(rate - previous_rate) / rate;
        const double attributable = std::max(0.0, relative_rate_change - rate_noise - previous_rate_noise);
        curvature = std::chrono::nanoseconds(static_cast<int64_t>(span_ns * attributable / 2.0));
    }

    const double closing_host_ns = static_cast<double>(closing.host.time_since_epoch().count());
    return ChordMapping{
        .mapping =
            experimental::ProgramRealtimeClockSync{
                .device_cycle_offset = std::llround(static_cast<double>(closing.ticks) - rate * closing_host_ns),
                .sync_error = interpolation_error(open, closing) + curvature,
            },
        .frequency = rate,
        .rate_noise = rate_noise,
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
        if (mapped_clock_lo_ != nullptr) {
            {  // latency-critical
                host_before = std::chrono::steady_clock::now();
                lo = *mapped_clock_lo_;
                host_after = std::chrono::steady_clock::now();
            }

            if (wrap_could_have_been_missed || lo < last_clock_lo_) {
                cached_clock_hi_ = *mapped_clock_hi_;
            }
        } else {
            auto& cluster = MetalContext::instance(context_id_).get_cluster();
            const tt_cxy_pair target(chip_id_, profiler_core_virtual_);
            host_before = std::chrono::steady_clock::now();
            cluster.read_reg(&lo, target, wall_clock_addr_lo_);
            host_after = std::chrono::steady_clock::now();
            if (wrap_could_have_been_missed || lo < last_clock_lo_) {
                cluster.read_reg(&cached_clock_hi_, target, wall_clock_addr_hi_);
            }
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
    if (!p.has_value()) {
        cost_.busy += std::chrono::steady_clock::now() - started_at;
        return false;
    }
    probes_.push_back(Anchor{p->host_time + placement_error(p->bracket), p->device_ticks, p->bracket});
    cost_.busy += std::chrono::steady_clock::now() - started_at;
    return true;
}

}  // namespace tt::tt_metal
