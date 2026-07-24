// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/realtime_profiler_clock_sync.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_metal.hpp>

#include "context/metal_context.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

namespace {

int64_t host_timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Re-anchor cadence. The init fit is taken at idle; under load the device clock drifts ~9ppm (thermal/DVFS), so the
// fixed slope goes stale — re-anchoring the offset this often bounds the accrued error to ~2µs.
constexpr auto kServoInterval = std::chrono::milliseconds(50);

// Round-trip busy-poll backstop. The first kRttProbeHealthyPolls reads skip the deadline check so a healthy handshake
// never reads the clock inside the round trip it is timing; only a stalled device reaches the check.
constexpr double kRttProbeTimeoutNs = 300'000.0;
constexpr uint32_t kRttProbeHealthyPolls = 128;

// Past this half-RTT a re-anchor's placement error exceeds the drift it would correct, so carry the previous anchor
// forward — unless it has itself gone stale (kMaxReanchorStaleness), where a loose anchor beats unbounded drift.
constexpr int64_t kMaxReanchorHalfRttNs = 20'000;
constexpr auto kMaxReanchorStaleness = std::chrono::seconds(2);

constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

// An overloaded host times out the round-trip probe every servo tick (kServoInterval) on every device at once, so
// warn at most this often per device and fold the rest into a suppressed count.
constexpr auto kProbeTimeoutWarnInterval = std::chrono::seconds(30);

// Host ACK buffer, 32-bit words: [token, device_time_lo, device_time_hi]. device_time is word 1 (offset 4) so its L1
// source is 4-mod-16: NOC PCIe writes require src/dst to share the low 4 bits, and the token source is 16-aligned.
constexpr uint32_t kSyncAckWords = 3;
constexpr uint32_t kSyncAckTokenWord = 0;

}  // namespace

std::optional<RealtimeProfilerClockFit> RealtimeProfilerCalibrationCache::try_get(
    uint32_t chip_id, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration max_age) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = by_chip_.find(chip_id);
    if (it != by_chip_.end() && now - it->second.updated_at < max_age) {
        return it->second.fit;
    }
    return std::nullopt;
}

void RealtimeProfilerCalibrationCache::put(
    uint32_t chip_id, const RealtimeProfilerClockFit& fit, std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    by_chip_[chip_id] = Entry{fit, now};
}

RealtimeProfilerCalibrationCache& rt_profiler_calibration_cache() {
    static RealtimeProfilerCalibrationCache cache;
    return cache;
}

void RealtimeProfilerClockSync::configure(
    ContextId context_id,
    IDevice* device,
    uint32_t chip_id,
    const CoreCoord& profiler_core,
    const MeshCoordinate& mesh_coord,
    bool hugepage_fallback,
    uint32_t sync_host_ts_addr,
    const std::shared_ptr<MeshDevice>& mesh_device,
    uint32_t ack_enc_addr,
    uint32_t ack_lo_addr,
    uint32_t ack_hi_addr) {
    context_id_ = context_id;
    device_ = device;
    chip_id_ = chip_id;
    profiler_core_ = profiler_core;
    mesh_coord_ = mesh_coord;
    hugepage_fallback_ = hugepage_fallback;
    sync_host_ts_addr_ = sync_host_ts_addr;
    configure_write_path();
    configure_ack_word(mesh_device, ack_enc_addr, ack_lo_addr, ack_hi_addr);
}

void RealtimeProfilerClockSync::configure_write_path() {
    if (MetalContext::instance(context_id_).hal().get_arch() != tt::ARCH::BLACKHOLE) {
        return;
    }
    try {
        const CoreCoord rt_virtual = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(device_->id())->get_tlb_manager();
        if (tlb_manager != nullptr) {
            sync_tlb_ = tlb_manager->get_tlb_window(tt_xy_pair(rt_virtual.x, rt_virtual.y));
        }
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: no TLB window for the profiler core ({}); sync uses write_core_immediate",
            device_->id(),
            e.what());
    }
}

void RealtimeProfilerClockSync::configure_ack_word(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t enc_addr, uint32_t lo_addr, uint32_t hi_addr) {
    const uint32_t device_id = device_->id();
    const auto write_field = [&](uint32_t addr, uint32_t val) {
        std::vector<uint32_t> data = {val};
        tt::tt_metal::detail::WriteToDeviceL1(device_, profiler_core_, addr, data, CoreType::WORKER);
    };
    try {
        if (hugepage_fallback_) {
            auto [ack_host, ack_dev_addr] = device_->sysmem_manager().allocate_region(kSyncAckWords * sizeof(uint32_t));
            if (ack_host == nullptr) {
                return;
            }
            ack_host_ptr_ = static_cast<volatile uint32_t*>(ack_host);
            for (uint32_t w = 0; w < kSyncAckWords; ++w) {
                const_cast<uint32_t*>(ack_host_ptr_)[w] = 0;
            }

            const auto& cluster = MetalContext::instance(context_id_).get_cluster();
            const auto& hal = MetalContext::instance(context_id_).hal();
            const auto& soc = cluster.get_soc_desc(cluster.get_associated_mmio_device(device_id));
            const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
            TT_FATAL(!pcie_cores.empty(), "No PCIe core found for RT-profiler sync ACK");
            write_field(enc_addr, hal.noc_xy_pcie64_encoding(pcie_cores.front().x, pcie_cores.front().y));
            write_field(lo_addr, ack_dev_addr);
            write_field(hi_addr, 0);
        } else {
            const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            std::shared_ptr<uint32_t[]> backing(
                static_cast<uint32_t*>(std::aligned_alloc(page, page)), [](uint32_t* p) { std::free(p); });
            if (!backing) {
                return;
            }
            for (uint32_t w = 0; w < kSyncAckWords; ++w) {
                backing[w] = 0;
            }
            tt::tt_metal::HostBuffer view(
                ttsl::Span<uint32_t>(backing.get(), kSyncAckWords), tt::tt_metal::MemoryPin(backing));
            MeshCoordinateRangeSet range;
            range.merge(MeshCoordinateRange(mesh_coord_));
            auto pinned = tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, range, view, true);
            if (!pinned) {
                return;
            }
            const auto noc = pinned->get_noc_addr(device_id);
            if (!noc.has_value()) {
                return;
            }
            ack_host_backing_ = backing;
            ack_pinned_ = pinned;
            ack_host_ptr_ = backing.get();
            write_field(enc_addr, noc->pcie_xy_enc);
            write_field(lo_addr, static_cast<uint32_t>(noc->addr & 0xFFFFFFFFull));
            write_field(hi_addr, static_cast<uint32_t>(noc->addr >> 32));
        }
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: host-ACK setup failed ({}); sync round-trip bound disabled",
            device_id,
            e.what());
    }
}

void RealtimeProfilerClockSync::write_timestamp(uint32_t value) {
    if (sync_tlb_ != nullptr) {
        sync_tlb_->write32(sync_host_ts_addr_, value);
        tt_driver_atomics::sfence();
    } else {
        const CoreCoord vcore = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
        MetalContext::instance(context_id_)
            .get_cluster()
            .write_core_immediate(&value, sizeof(value), tt_cxy_pair(device_->id(), vcore), sync_host_ts_addr_);
        tt_driver_atomics::sfence();
    }
}

uint32_t RealtimeProfilerClockSync::read_ack() const {
#if defined(__x86_64__) || defined(__i386__)
    // Hugepage fallback: device PCIe writes may be non-snooped; evict the line to avoid reading a stale copy.
    if (hugepage_fallback_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(ack_host_ptr_)));
        _mm_lfence();
    }
#endif
    return ack_host_ptr_[kSyncAckTokenWord];
}

uint64_t RealtimeProfilerClockSync::read_device_time() const {
#if defined(__x86_64__) || defined(__i386__)
    if (hugepage_fallback_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(ack_host_ptr_)));
        _mm_lfence();
    }
#endif
    // device_time (words 1,2) is NOC-written before the token, so it is valid once read_ack has matched.
    return (static_cast<uint64_t>(ack_host_ptr_[2]) << 32) | static_cast<uint64_t>(ack_host_ptr_[1]);
}

int64_t RealtimeProfilerClockSync::measure_rtt(int64_t host_before, uint32_t host_time_id) {
    if (ack_host_ptr_ == nullptr) {
        return 0;
    }
    const int64_t deadline = host_before + static_cast<int64_t>(kRttProbeTimeoutNs);
    uint32_t polls = 0;
    while (read_ack() != host_time_id) {
        if (++polls > kRttProbeHealthyPolls && host_timestamp_ns() > deadline) {
            return -1;
        }
    }
    return host_timestamp_ns() - host_before;
}

std::optional<RealtimeProfilerClockSync::Probe> RealtimeProfilerClockSync::probe() {
    const int64_t host_before = host_timestamp_ns();
    if (++sync_seq_ == 0) {
        sync_seq_ = 1;
    }
    const uint32_t host_time_id = sync_seq_;
    write_timestamp(host_time_id);
    const int64_t rtt_ticks = measure_rtt(host_before, host_time_id);
    if (rtt_ticks < 0) {
        return std::nullopt;
    }
    const uint64_t device_time = read_device_time();
    return Probe{host_before, rtt_ticks, device_time};
}

void RealtimeProfilerClockSync::reanchor(int64_t host_anchor, uint64_t device_anchor) {
    device_cycle_offset_ =
        std::llround(static_cast<double>(device_anchor) - frequency_ * static_cast<double>(host_anchor));
}

bool RealtimeProfilerClockSync::servo_due(std::chrono::steady_clock::time_point now) const {
    return !last_reanchor_at_.has_value() || now - *last_reanchor_at_ >= kServoInterval;
}

void RealtimeProfilerClockSync::fit_from_samples(
    const std::vector<Probe>& samples, int64_t host_start_time, double default_frequency) {
    uint64_t device_anchor = 0;
    if (samples.size() >= 2) {
        const double n = static_cast<double>(samples.size());
        double host_mean = 0.0;
        double device_mean = 0.0;
        for (const auto& s : samples) {
            host_mean += static_cast<double>(s.host_before - host_start_time);
            device_mean += static_cast<double>(s.device_time);
        }
        host_mean /= n;
        device_mean /= n;

        double num = 0.0;
        double den = 0.0;
        for (const auto& s : samples) {
            double dx = static_cast<double>(s.host_before - host_start_time) - host_mean;
            double dy = static_cast<double>(s.device_time) - device_mean;
            num += dx * dy;
            den += dx * dx;
        }
        frequency_ = (std::abs(den) > 1e-10) ? num / den : default_frequency;

        // Intercept via means: intercept = ȳ - slope * x̄ = device cycle count at host_time = 0.
        const double intercept = device_mean - frequency_ * host_mean;
        device_anchor = static_cast<uint64_t>(intercept);

        double residual_sumsq_ns = 0.0;
        double residual_max_ns = 0.0;
        for (const auto& s : samples) {
            const double predicted_device =
                device_mean + frequency_ * (static_cast<double>(s.host_before - host_start_time) - host_mean);
            const double residual_ns = (static_cast<double>(s.device_time) - predicted_device) / frequency_;
            residual_sumsq_ns += residual_ns * residual_ns;
            residual_max_ns = std::max(residual_max_ns, std::abs(residual_ns));
        }
        const double residual_rms_ns = std::sqrt(residual_sumsq_ns / n);

        log_info(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "device_time_at_sync={} cycles, fit residual rms={:.0f} ns max={:.0f} ns",
            chip_id_,
            samples.size(),
            frequency_,
            device_anchor,
            residual_rms_ns,
            residual_max_ns);
    } else {
        frequency_ = default_frequency;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using default frequency",
            chip_id_);
    }
    reanchor(host_start_time, device_anchor);
}

bool RealtimeProfilerClockSync::run_fit(uint32_t num_samples) {
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr uint32_t kRunSyncMaxConsecutiveTimeouts = 3;
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const int64_t host_start_time = host_timestamp_ns();

    std::vector<Probe> samples;
    if (ack_host_ptr_ == nullptr) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} has no host ACK word; skipping sync (using default frequency)",
            chip_id_);
    } else {
        std::this_thread::sleep_for(kRunSyncSettleDelay);

        uint32_t consecutive_timeouts = 0;
        for (uint32_t i = 0; i < num_samples + 1; i++) {
            std::this_thread::sleep_for(kRunSyncSampleInterval);

            const auto p = probe();
            if (!p.has_value()) {
                consecutive_timeouts++;
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} sync sample {}/{} round-trip probe timed out "
                    "(consecutive timeouts: {}/{})",
                    chip_id_,
                    i,
                    num_samples,
                    consecutive_timeouts,
                    kRunSyncMaxConsecutiveTimeouts);
                if (consecutive_timeouts >= kRunSyncMaxConsecutiveTimeouts) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {} sync aborted: {} consecutive timeouts. "
                        "Device kernel may not be responding (check DPRINT output).",
                        chip_id_,
                        consecutive_timeouts);
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            // Discard first sample - can be very off due to cold PCIe path.
            if (i == 0) {
                continue;
            }
            samples.push_back(*p);
        }
    }

    const double default_frequency = cluster.get_device_aiclk(chip_id_) / 1000.0;
    const bool fit_ok = samples.size() >= 2;
    fit_from_samples(samples, host_start_time, default_frequency);
    return fit_ok;
}

bool RealtimeProfilerClockSync::try_restore_from_cache(std::chrono::steady_clock::time_point now) {
    auto fit = rt_profiler_calibration_cache().try_get(chip_id_, now, kRtProfilerMinSyncInterval);
    if (!fit.has_value()) {
        return false;
    }
    frequency_ = fit->frequency;
    device_cycle_offset_ = fit->device_cycle_offset;
    rtt_ticks_ = fit->rtt_ticks;
    last_reanchor_at_ = now;
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device {}: reusing cached calibration (last sync within {}s), skipping init sync",
        chip_id_,
        static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(kRtProfilerMinSyncInterval).count()));
    return true;
}

void RealtimeProfilerClockSync::service_servo(std::chrono::steady_clock::time_point now) {
    if (ack_host_ptr_ == nullptr) {
        return;
    }
    if (!servo_due(now)) {
        return;
    }
    try {
        const auto p = probe();
        if (!p.has_value()) {
            if (now - last_probe_timeout_warn_ >= kProbeTimeoutWarnInterval) {
                if (suppressed_probe_timeouts_ > 0) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {} sync round-trip probe timed out; keeping previous mapping "
                        "({} further probe timeouts suppressed)",
                        chip_id_,
                        suppressed_probe_timeouts_);
                } else {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {} sync round-trip probe timed out; keeping previous mapping",
                        chip_id_);
                }
                last_probe_timeout_warn_ = now;
                suppressed_probe_timeouts_ = 0;
            } else {
                ++suppressed_probe_timeouts_;
            }
            return;
        }
        suppressed_probe_timeouts_ = 0;
        const bool prev_anchor_fresh =
            last_reanchor_at_.has_value() && now - *last_reanchor_at_ < kMaxReanchorStaleness;
        if (p->rtt_ticks / 2 > kMaxReanchorHalfRttNs && prev_anchor_fresh) {
            return;
        }
        rtt_ticks_ = p->rtt_ticks;
        reanchor(p->host_before + p->rtt_ticks / 2, p->device_time);
        last_reanchor_at_ = now;
        rt_profiler_calibration_cache().put(
            chip_id_, RealtimeProfilerClockFit{frequency_, device_cycle_offset_, rtt_ticks_}, now);
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[Real-time profiler] Failed to start sync for device {}: {}", chip_id_, e.what());
    }
}

experimental::ProgramRealtimeClockSync RealtimeProfilerClockSync::mapping() const {
    const uint64_t sync_error_ns = static_cast<uint64_t>(std::llround(static_cast<double>(rtt_ticks_) * 0.5));
    return experimental::ProgramRealtimeClockSync{device_cycle_offset_, sync_error_ns};
}

}  // namespace tt::tt_metal::distributed
