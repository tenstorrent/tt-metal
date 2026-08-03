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

// Sync policy. A least squares slope over a span T with N probes of placement noise s carries a relative frequency
// error of s*sqrt(12)/(T*sqrt(N)), and that error only accrues between anchors, contributing error_ppm *
// kSyncInterval to the mapping. Shortening the fit span is paid for out of that margin.

// Bring-up fit: kFitProbes points spaced kProbeInterval apart, each the tightest of kProbesPerPoint reads, after
// kSettleDelay of letting AICLK settle. ~0.5s per device at these values.
constexpr uint32_t kFitProbes = 100;
constexpr auto kProbeInterval = std::chrono::milliseconds(5);
constexpr auto kSettleDelay = std::chrono::milliseconds(50);
// One preempted read has a bracket wide enough that its midpoint is microseconds out, and at the ends of the span
// that tilts the slope by tens of ppm.
constexpr int kProbesPerPoint = 4;

constexpr auto kCalibrationCacheMaxAge = std::chrono::seconds(60);

// The counter could have been read anywhere inside the bracket.
constexpr std::chrono::nanoseconds placement_error(std::chrono::nanoseconds bracket) { return bracket / 2; }

// How far above the median a bring-up probe's bracket may sit and still be regressed. Loose enough that ordinary
// spread survives and only the reads serviced late are cut.
constexpr int kFitBracketOutlierFactor = 2;

// Lets a rapid MeshDevice reopen skip the bring-up fit and take one anchor probe instead; device WALL_CLOCK
// free-runs across close. The offset is not cached, since it is re-anchored every kSyncInterval anyway.
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
}

void RealtimeProfilerClockModel::set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks) {
    const double host_ns = static_cast<double>(host_time.time_since_epoch().count());
    device_cycle_offset_ = std::llround(static_cast<double>(device_ticks) - frequency_ * host_ns);
}

std::optional<RealtimeProfilerClockModel::FitResidual> RealtimeProfilerClockModel::fit(
    std::span<const ClockProbe> probes, std::chrono::steady_clock::time_point host_start) {
    if (probes.size() < 2) {
        // A slope needs two, but the offset needs only one and the seeded frequency is already a usable slope.
        if (probes.size() == 1) {
            try_reanchor(probes.front());
        }
        return std::nullopt;
    }

    // A probe serviced late sits further right than its pairing implies, so regressed with equal weight a handful of
    // them tilt the slope by tens of ppm -- which the servo then spends the session correcting as drift.
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
    // If filtering leaves too few to regress, the scatter is what the link is doing; fit everything and let the
    // residual report it rather than inventing a slope from two points.
    const std::span<const ClockProbe> fitted_probes = tight.size() >= 2 ? std::span<const ClockProbe>(tight) : probes;

    // Mean-centered on host_start: regressing at absolute-timestamp magnitudes loses most of the significant digits
    // of the sums to catastrophic cancellation. Centered at the same instant try_reanchor anchors at, so the initial
    // anchor is not offset from every later one by half a bracket.
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

    // Intercept via means: the device tick count at centered host time 0, i.e. at host_start.
    set_anchor(host_start, static_cast<uint64_t>(device_mean - frequency_ * host_mean));

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

    // Without these mapping() would report zero error until the first resync. The median over every offered probe,
    // not just the regressed ones, keeps it the conservative of the two.
    bracket_ = median_bracket;
    residual_ = placement_error(median_bracket);
    last_reanchor_at_ = probes.back().host_time;
    return residual;
}

std::chrono::nanoseconds RealtimeProfilerClockModel::drift_at(const ClockProbe& probe) const {
    if (!last_reanchor_at_.has_value()) {
        return {};
    }
    const double at_ns =
        static_cast<double>((probe.host_time + placement_error(probe.bracket)).time_since_epoch().count());
    const double predicted = frequency_ * at_ns + static_cast<double>(device_cycle_offset_);
    return std::chrono::nanoseconds(
        std::llround(std::abs(static_cast<double>(probe.device_ticks) - predicted) / frequency_));
}

bool RealtimeProfilerClockModel::try_reanchor(const ClockProbe& probe) {
    if (last_reanchor_at_.has_value()) {
        // How far the standing mapping actually missed this probe. This is the mapping's real error, measured, and
        // it is what the decision below is made on -- no assumed drift rate enters anywhere.
        last_drift_ = drift_at(probe);

        // A probe cannot locate the clock better than half its own bracket, so a miss smaller than that says nothing:
        // re-anchoring on it would trade a good anchor for a noisier one. Take the probe when it would leave the
        // mapping better placed than the miss it just measured, or than the standing anchor is placed -- the second
        // clause is free and keeps the published error from being stuck at whatever a dip forced us to accept.
        if (placement_error(probe.bracket) >= last_drift_ &&
            placement_error(probe.bracket) >= placement_error(bracket_)) {
            // The mapping is no better than where its anchor was placed, however small the miss just measured.
            residual_ = std::max(placement_error(bracket_), last_drift_);
            return false;
        }
    }
    bracket_ = probe.bracket;
    residual_ = placement_error(probe.bracket);
    set_anchor(probe.host_time + placement_error(probe.bracket), probe.device_ticks);
    last_reanchor_at_ = probe.host_time;
    return true;
}

experimental::ProgramRealtimeClockSync RealtimeProfilerClockModel::mapping() const {
    return experimental::ProgramRealtimeClockSync{
        .device_cycle_offset = device_cycle_offset_,
        // What the last probe found, not a modelled rate: after a re-anchor this is where the new anchor was
        // placed, and otherwise it is the miss the mapping is currently carrying.
        .sync_error = residual_,
    };
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
        // Resolved once: TlbWindow's accessors are virtual calls that re-validate the offset on every access.
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

std::optional<ClockProbe> RealtimeProfilerClockSync::probe() {
    // Opened before the bracket so its own cost stays outside it. Nothing between the two clock reads may be
    // instrumented, for the same reason.
    TTZoneScopedDN(RT_PROFILER, "Probe");
    uint32_t lo = 0;
    try {
        // The low word going backwards only reveals a wrap while at most one has happened, so past that the high
        // word has to be re-read outright. Halved for margin against a badly seeded frequency.
        const auto wrap_period =
            std::chrono::nanoseconds(static_cast<int64_t>((1ull << 32) / std::max(model_.frequency(), 0.1)));
        const bool wrap_could_have_been_missed = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                                                 std::chrono::steady_clock::now() - last_probe_at_ > wrap_period / 2;

        std::chrono::steady_clock::time_point host_before;
        std::chrono::steady_clock::time_point host_after;
        if (mapped_clock_lo_ != nullptr) {
            host_before = std::chrono::steady_clock::now();
            lo = *mapped_clock_lo_;
            host_after = std::chrono::steady_clock::now();
            // A read of the low word latches the high word, so this yields the half completing the bracket's value
            // and can sit outside the bracket. Re-reading also recovers from any number of missed wraps.
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
    resync();
    if (!model_.is_anchored()) {
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
    // A settled clock fits to ~2ns rms. A fit taken across an AICLK ramp still looks well-conditioned but lands tens
    // of ppm off, showing up as a residual three orders of magnitude above that.
    constexpr double kMaxFitResidualRmsNs = 200.0;
    const auto host_start_time = std::chrono::steady_clock::now();

    std::vector<ClockProbe> probes;
    probes.reserve(kFitProbes);
    std::this_thread::sleep_for(kSettleDelay);

    // The cold PCIe path would otherwise land at one end of the span, where a badly placed point has the most
    // leverage over the slope.
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
    // Unconditional: records can be drained before bring-up finishes, so a mapping has to be readable even when the
    // fit is judged below not worth keeping.
    if (!residual.has_value()) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough probes, using the commanded AICLK",
            chip_id_);
        return false;
    }

    // Against kFitProbes rather than what was collected, so it catches both a link that could not be read and one
    // whose probes the model cut. Half still fits a slope well -- the span is unchanged and the count only enters
    // under a square root -- but below that another pass is likely to beat this one.
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
    const auto p = best_of(kResyncProbes);
    if (!p.has_value()) {
        return false;
    }
    model_.try_reanchor(*p);
    return true;
}

}  // namespace tt::tt_metal
