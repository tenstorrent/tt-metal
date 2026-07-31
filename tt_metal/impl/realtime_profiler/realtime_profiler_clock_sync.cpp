// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tt-logger/tt-logger.hpp>

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

// How long a cached calibration stays usable across a MeshDevice close/reopen.
constexpr auto kCalibrationCacheMaxAge = std::chrono::seconds(60);

// Process-global per-physical-chip cache of the fitted clock frequency, so a rapid MeshDevice reopen can skip the
// ~0.5s bring-up fit and take one anchor probe instead (device WALL_CLOCK is free-running across close). The offset is
// not cached: it is re-anchored every kClockSyncInterval, so a stored one would be stale before first use.
class RealtimeProfilerFrequencyCache {
public:
    std::optional<double> try_get(
        uint32_t chip_id,
        std::chrono::steady_clock::time_point now,
        std::chrono::steady_clock::duration max_age) const {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = by_chip_.find(chip_id);
        if (it != by_chip_.end() && now - it->second.updated_at < max_age) {
            return it->second.frequency;
        }
        return std::nullopt;
    }

    void put(uint32_t chip_id, double frequency, std::chrono::steady_clock::time_point now) {
        std::lock_guard<std::mutex> lock(mu_);
        by_chip_[chip_id] = Entry{frequency, now};
    }

private:
    struct Entry {
        double frequency = 0.0;
        std::chrono::steady_clock::time_point updated_at;
    };
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, Entry> by_chip_;
};

RealtimeProfilerFrequencyCache& rt_profiler_frequency_cache() {
    static RealtimeProfilerFrequencyCache cache;
    return cache;
}

}  // namespace

RealtimeProfilerClockSync::RealtimeProfilerClockSync() = default;
RealtimeProfilerClockSync::~RealtimeProfilerClockSync() = default;

void RealtimeProfilerClockSync::configure(const RealtimeProfilerClockSyncConfig& config) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    context_id_ = config.context_id;
    device_ = config.device;
    chip_id_ = config.device->id();
    profiler_core_ = config.profiler_core;
    profiler_core_virtual_ = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
    const auto& hal = MetalContext::instance(context_id_).hal();
    wall_clock_addr_lo_ = hal.get_tensix_wall_clock_reg_addr_lo();
    wall_clock_addr_hi_ = hal.get_tensix_wall_clock_reg_addr_hi();
    // A chip always has a usable frequency, whatever happens to its sync. Later steps only refine it.
    model_.seed_frequency(MetalContext::instance(context_id_).get_cluster().get_device_aiclk(chip_id_) / 1000.0);
    configure_clock_read_path();
    // Records can be drained before the first sample is taken, so the seeded AICLK has to be readable already.
    publish_mapping();
}

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
        // Resolved once: TlbWindow's accessors are virtual calls that re-validate the offset and re-derive the address
        // on every access, and the window is fixed, so the address is not.
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

void RealtimeProfilerClockSync::publish_mapping() {
    const experimental::ProgramRealtimeClockSync mapping = model_.mapping();
    const uint32_t seq = mapping_seq_.load(std::memory_order_relaxed);
    mapping_seq_.store(seq + 1, std::memory_order_relaxed);  // odd: an update is in progress
    std::atomic_thread_fence(std::memory_order_release);
    mapping_device_cycle_offset_.store(mapping.device_cycle_offset, std::memory_order_relaxed);
    mapping_sync_error_.store(mapping.sync_error, std::memory_order_relaxed);
    mapping_frequency_.store(model_.frequency(), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    mapping_seq_.store(seq + 2, std::memory_order_release);
}

RealtimeProfilerClockSync::Calibration RealtimeProfilerClockSync::calibration() const {
    while (true) {
        const uint32_t before = mapping_seq_.load(std::memory_order_acquire);
        if ((before & 1u) != 0u) {
            continue;  // caught the sync thread mid-update
        }
        Calibration out;
        out.mapping.device_cycle_offset = mapping_device_cycle_offset_.load(std::memory_order_relaxed);
        out.mapping.sync_error = mapping_sync_error_.load(std::memory_order_relaxed);
        out.frequency = mapping_frequency_.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (mapping_seq_.load(std::memory_order_relaxed) == before) {
            return out;
        }
    }
}

std::optional<ClockSyncSample> RealtimeProfilerClockSync::probe() {
    // Opened before the bracket so the zone's own cost stays outside it. Nothing between the two clock reads may be
    // instrumented, for the same reason.
    TTZoneScopedDN(RT_PROFILER, "Probe");
    uint32_t lo = 0;
    try {
        // The high word is re-read only when it could have changed: after a wrap, or after a gap long enough that a
        // wrap might have been missed entirely. A read of the low word latches the high word, so when it is read it
        // completes the value sampled inside the bracket and does not need to sit inside it.
        constexpr auto kMaxGapBeforeRereadingHigh = std::chrono::seconds(1);
        const bool reread_high = last_probe_at_ == std::chrono::steady_clock::time_point{} ||
                                 std::chrono::steady_clock::now() - last_probe_at_ > kMaxGapBeforeRereadingHigh;
        std::chrono::steady_clock::time_point host_before;
        std::chrono::steady_clock::time_point host_after;
        if (mapped_clock_lo_ != nullptr) {
            host_before = std::chrono::steady_clock::now();
            lo = *mapped_clock_lo_;
            host_after = std::chrono::steady_clock::now();
            if (reread_high) {
                cached_clock_hi_ = *mapped_clock_hi_;
            }
        } else {
            auto& cluster = MetalContext::instance(context_id_).get_cluster();
            const tt_cxy_pair target(chip_id_, profiler_core_virtual_);
            host_before = std::chrono::steady_clock::now();
            cluster.read_reg(&lo, target, wall_clock_addr_lo_);
            host_after = std::chrono::steady_clock::now();
            if (reread_high) {
                cluster.read_reg(&cached_clock_hi_, target, wall_clock_addr_hi_);
            }
        }
        if (!reread_high && lo < last_clock_lo_) {
            ++cached_clock_hi_;  // the low word wrapped since the previous probe
        }
        last_clock_lo_ = lo;
        last_probe_at_ = host_after;
        const uint32_t hi = cached_clock_hi_;
        const auto bracket = host_after - host_before;
        TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(bracket.count()));
        return ClockSyncSample{
            host_before,
            std::chrono::duration_cast<std::chrono::nanoseconds>(bracket),
            (static_cast<uint64_t>(hi) << 32) | lo};
    } catch (const std::exception& e) {
        log_debug(tt::LogMetal, "[Real-time profiler] Device {}: clock read failed ({})", chip_id_, e.what());
        return std::nullopt;
    }
}

std::optional<ClockSyncSample> RealtimeProfilerClockSync::best_of(int probes) {
    std::optional<ClockSyncSample> best;
    for (int i = 0; i < probes; i++) {
        const auto sample = probe();
        if (sample.has_value() && (!best.has_value() || sample->rtt < best->rtt)) {
            best = sample;
        }
    }
    return best;
}

bool RealtimeProfilerClockSync::calibrate() {
    TTZoneScopedDN(RT_PROFILER, "Calibrate");
    // Enough that the fitted slope is dominated by the baseline rather than per-sample noise. At 5ms spacing this is
    // ~0.5s of bring-up per device.
    constexpr uint32_t kFitSamples = 100;
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr uint32_t kRunSyncMaxConsecutiveFailures = 3;
    // Each fitted point is the tightest of a few reads, for the same reason resync bursts: one preempted read has a
    // bracket wide enough that its midpoint is microseconds out, and at the ends of the span that tilts the slope by
    // tens of ppm -- which the servo then spends the session correcting as though the chip were drifting.
    constexpr int kProbesPerSample = 4;
    // A settled clock fits this line to ~2ns rms; a fit taken while AICLK is ramping still uses every sample and still
    // looks well-conditioned, but lands a frequency tens of ppm off and shows up as a residual three orders of
    // magnitude above that. Retrying is cheap and the ramp is over in well under a second, where keeping the fit would
    // have the servo correcting the error as drift for the rest of the session.
    constexpr double kMaxFitResidualRmsNs = 200.0;
    const auto host_start_time = std::chrono::steady_clock::now();

    std::vector<ClockSyncSample> samples;
    samples.reserve(kFitSamples);
    std::this_thread::sleep_for(kRunSyncSettleDelay);

    // The first read pays the cold PCIe path, and sits at one end of the fitted span where a badly placed point has
    // the most leverage over the slope. Taken and dropped rather than skipped, so the path is warm for the rest.
    (void)probe();

    uint32_t consecutive_failures = 0;
    for (uint32_t attempt = 0; attempt < kFitSamples; attempt++) {
        std::this_thread::sleep_for(kRunSyncSampleInterval);
        const auto p = best_of(kProbesPerSample);
        if (!p.has_value()) {
            if (++consecutive_failures >= kRunSyncMaxConsecutiveFailures) {
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
        samples.push_back(*p);
    }

    // configure() already seeded the commanded AICLK, which the model keeps if the fit has too few samples. The model
    // discards the loose brackets among these before regressing.
    const std::optional<ClockModel::FitResidual> residual = model_.fit(samples, host_start_time);
    // Unconditional: records can be drained before calibration finishes, so a mapping has to be readable even when
    // the fit is judged below not worth keeping.
    publish_mapping();
    if (!residual.has_value()) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using the commanded AICLK",
            chip_id_);
        return false;
    }

    // Measured against kFitSamples rather than what was collected, so this catches both a link that could not be read
    // and one whose samples were scattered enough for the model to cut them. Half still fits a slope well -- the span
    // is unchanged and the count only enters under a square root -- but below that another pass is likely to beat this
    // one. Returning false re-runs the whole calibration through calibrate_device's existing retry budget, which is
    // what bounds the cost of a chip that keeps failing.
    if (residual->num_samples_fitted * 2 < kFitSamples) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} fit only {} of {} wanted sync samples; retrying rather than fitting a "
            "frequency from what is left",
            chip_id_,
            residual->num_samples_fitted,
            kFitSamples);
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

    // Cached only once the fit is worth reusing: try_restore_calibration hands the cached frequency to a later
    // MeshDevice without re-fitting, so a bad one would outlive the run that produced it.
    rt_profiler_frequency_cache().put(chip_id_, model_.frequency(), std::chrono::steady_clock::now());
    log_info(
        tt::LogMetal,
        "[Real-time profiler] Device {} sync complete: fit {} of {} collected samples, frequency={:.6f} GHz, fit "
        "residual rms={:.0f} ns max={:.0f} ns",
        chip_id_,
        residual->num_samples_fitted,
        residual->num_samples_offered,
        model_.frequency(),
        residual->rms_ns,
        residual->max_ns);
    return true;
}

bool RealtimeProfilerClockSync::try_restore_calibration(std::chrono::steady_clock::time_point now) {
    // Present or absent, this zone says which bring-up path the device took: the cached-frequency restore or the
    // full fit below it.
    TTZoneScopedDN(RT_PROFILER, "RestoreCalibration");
    const auto frequency = rt_profiler_frequency_cache().try_get(chip_id_, now, kCalibrationCacheMaxAge);
    if (!frequency.has_value()) {
        return false;
    }
    model_.seed_frequency(*frequency);
    publish_mapping();
    resync();
    if (!model_.is_anchored()) {
        return false;  // the read failed, so fall back to a full fit
    }
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device {}: reusing cached clock frequency (fit within {}s), skipping the multi-sample "
        "fit",
        chip_id_,
        static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(kCalibrationCacheMaxAge).count()));
    return true;
}

bool RealtimeProfilerClockSync::resync() {
    TTZoneScopedDN(RT_PROFILER, "Resync");
    // The bracket is what bounds the anchor, so the tightest of a few reads is the best anchor available. Ranked
    // rather than compared against a threshold: under record load the whole distribution shifts, and an absolute
    // threshold would reject every sample in a pass instead of picking that pass's best. Four is where the returns
    // stop: the bracket distribution is tight enough (min 690ns, p99 740ns) that the tightest of eight beats the
    // tightest of four by ~3ns of published error, for 1.6x the host cost.
    constexpr int kProbes = 4;
    const auto best = best_of(kProbes);
    if (!best.has_value()) {
        return false;
    }
    if (model_.try_reanchor(*best)) {
        publish_mapping();
    }
    return true;
}

}  // namespace tt::tt_metal
