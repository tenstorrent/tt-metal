// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <memory>
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
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_metal.hpp>

#include "context/metal_context.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

// How far above the standing anchor a probe may land and still end the burst. Comparing for strict improvement makes
// the threshold converge on the fastest round trip ever measured, after which no probe can pass it and every resync
// pays the full depth to shave nanoseconds the published bound does not notice. 1/32 measured against 1/64 and 1/16:
// it takes 2.6 probes to strict comparison's 6.4, for 10 ns on the median bound and 30 ns on its 99th percentile.
constexpr int kResyncExitToleranceDivisor = 32;
constexpr std::chrono::nanoseconds resync_exit_threshold(std::chrono::nanoseconds anchor_rtt) {
    return anchor_rtt + anchor_rtt / kResyncExitToleranceDivisor;
}

// Round-trip busy-poll backstop, not an accept threshold. The first kRttProbeHealthyPolls reads skip the deadline
// check so a healthy handshake never reads the clock inside the round trip it is timing; only a stalled device
// reaches the check.
//
// Steady state keeps a handshake only if it places the anchor better than the standing one has since degraded, so at
// a ~1.3us round trip nothing above a few microseconds is ever taken. Polling far past that spends the sync thread on
// a sample destined to be discarded, and the cost lands exactly when a device stops answering: every probe then runs
// to full depth, and a pass over 32 devices has to fit inside kClockSyncInterval. The bound below still leaves room
// for the drift term to justify a slower handshake after a couple of seconds of rejections, which is as long as this
// is worth tolerating before the stall is the real problem.
constexpr auto kResyncProbeTimeout = std::chrono::microseconds(50);
// Bring-up has no standing anchor to beat, so its first handshake is accepted however slow. Worth waiting out a
// loaded machine rather than giving up and running on the seeded AICLK.
constexpr auto kCalibrateProbeTimeout = std::chrono::microseconds(300);
constexpr uint32_t kRttProbeHealthyPolls = 128;

// How long a cached calibration stays usable across a MeshDevice close/reopen.
constexpr auto kCalibrationCacheMaxAge = std::chrono::seconds(60);

// Host ACK buffer, 32-bit words: [device_time_lo, device_time_hi, token]. device_time is at the base so it is 8-byte
// aligned; NOC PCIe writes require src/dst to share the low 4 bits, so its L1 source is 16-aligned and the token's is
// 8-mod-16.
constexpr uint32_t kSyncAckWords = 3;
constexpr uint32_t kSyncAckTokenWord = 2;

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

void RealtimeProfilerClockSync::configure(const RealtimeProfilerClockSyncConfig& config) {
    TTZoneScopedDN(RT_PROFILER, "ClockSyncConfigure");
    context_id_ = config.context_id;
    device_ = config.device;
    chip_id_ = config.device->id();
    profiler_core_ = config.profiler_core;
    mesh_coord_ = config.mesh_coord;
    hugepage_fallback_ = config.hugepage_fallback;
    l1_ = resolve_l1_addrs(config.msg_base_addr);
    // A chip always has a usable frequency, whatever happens to its sync handshake; later steps only refine it.
    model_.seed_frequency(MetalContext::instance(context_id_).get_cluster().get_device_aiclk(chip_id_) / 1000.0);
    configure_write_path();
    configure_ack_word(*config.mesh_device);
    // Records can be drained before the first handshake completes, so the seeded AICLK has to be readable already.
    publish_mapping();
}

void RealtimeProfilerClockSync::publish_mapping() {
    const experimental::ProgramRealtimeClockSync mapping = model_.mapping();
    const uint32_t seq = mapping_seq_.load(std::memory_order_relaxed);
    mapping_seq_.store(seq + 1, std::memory_order_relaxed);  // odd: an update is in progress
    std::atomic_thread_fence(std::memory_order_release);
    mapping_device_cycle_offset_.store(mapping.device_cycle_offset, std::memory_order_relaxed);
    mapping_sync_error_ns_.store(mapping.sync_error_ns, std::memory_order_relaxed);
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
        out.mapping.sync_error_ns = mapping_sync_error_ns_.load(std::memory_order_relaxed);
        out.frequency = mapping_frequency_.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (mapping_seq_.load(std::memory_order_relaxed) == before) {
            return out;
        }
    }
}

RealtimeProfilerClockSync::SyncL1Addrs RealtimeProfilerClockSync::resolve_l1_addrs(uint32_t msg_base_addr) const {
    using Msg = realtime_profiler_msgs::realtime_profiler_msg_t;
    const auto& factory =
        MetalContext::instance(context_id_).hal().get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    const auto field_addr = [&](Msg::Field field) -> uint32_t {
        return msg_base_addr + static_cast<uint32_t>(factory.offset_of<Msg>(field));
    };
    return SyncL1Addrs{
        .token = field_addr(Msg::Field::sync_token),
        .ack_host_addr = field_addr(Msg::Field::sync_ack_host_addr),
    };
}

void RealtimeProfilerClockSync::configure_write_path() {
    try {
        const CoreCoord rt_virtual = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
        const tt_xy_pair tlb_core(rt_virtual.x, rt_virtual.y);
        auto* tlb_manager =
            MetalContext::instance(context_id_).get_cluster().get_driver()->get_chip(device_->id())->get_tlb_manager();
        // ll_api::configure_static_tlbs gives every TENSIX core a static window at cluster init, keyed by its
        // TRANSLATED coordinate, which is exactly what virtual_core_from_logical_core returns for a WORKER. The
        // profiler core is a TENSIX worker, so this lookup hits its own window on every architecture that reaches
        // here. is_tlb_mapped is the precondition get_tlb_window would otherwise throw on, and bounds-checks the
        // token against the window; nothing is allocated, the window already exists.
        if (tlb_manager == nullptr || !tlb_manager->is_tlb_mapped(tlb_core, l1_.token, sizeof(uint32_t))) {
            return;
        }
        sync_tlb_ = tlb_manager->get_tlb_window(tlb_core);
        // Resolve the token's mapped address once. TlbWindow::write32 is a virtual call that re-validates the
        // offset and re-derives the address on every store; the window is static, so the address is not.
        const uint64_t window_offset =
            sync_tlb_->get_base_address() - sync_tlb_->handle_ref().get_config().local_offset;
        sync_doorbell_ =
            reinterpret_cast<volatile uint32_t*>(sync_tlb_->handle_ref().get_base() + l1_.token + window_offset);
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: no TLB window for the profiler core ({}); sync uses write_core_immediate",
            device_->id(),
            e.what());
    }
}

void RealtimeProfilerClockSync::configure_ack_word(distributed::MeshDevice& mesh_device) {
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

            write_field(l1_.ack_host_addr, ack_dev_addr);
            write_field(l1_.ack_host_addr + sizeof(uint32_t), 0);
        } else {
            const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            // Page-aligned because PinnedMemory maps whole pages; aligned new rather than aligned_alloc so the
            // deleter is not hand-rolled memory management.
            std::shared_ptr<uint32_t[]> backing(
                new (std::align_val_t{page}) uint32_t[page / sizeof(uint32_t)],
                [page](uint32_t* p) { operator delete[](p, std::align_val_t{page}); });
            for (uint32_t w = 0; w < kSyncAckWords; ++w) {
                backing[w] = 0;
            }
            tt::tt_metal::HostBuffer view(
                ttsl::Span<uint32_t>(backing.get(), kSyncAckWords), tt::tt_metal::MemoryPin(backing));
            distributed::MeshCoordinateRangeSet range;
            range.merge(distributed::MeshCoordinateRange(mesh_coord_));
            auto pinned = tt::tt_metal::experimental::PinnedMemory::Create(mesh_device, range, view, true);
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
            write_field(l1_.ack_host_addr, static_cast<uint32_t>(noc->addr & 0xFFFFFFFFull));
            write_field(l1_.ack_host_addr + sizeof(uint32_t), static_cast<uint32_t>(noc->addr >> 32));
        }
    } catch (const std::exception& e) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {}: host-ACK setup failed ({}); sync round-trip bound disabled",
            device_id,
            e.what());
    }
}

void RealtimeProfilerClockSync::write_token(uint32_t value) {
    if (sync_doorbell_ != nullptr) {
        *sync_doorbell_ = value;
        tt_driver_atomics::sfence();
    } else {
        // Resolves the virtual core inside the interval the round trip is timed over. Every chip that gives the
        // profiler core a static window takes the branch above, so this cost has nowhere to show up in practice.
        const CoreCoord vcore = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
        MetalContext::instance(context_id_)
            .get_cluster()
            .write_core_immediate(&value, sizeof(value), tt_cxy_pair(device_->id(), vcore), l1_.token);
        tt_driver_atomics::sfence();
    }
}

void RealtimeProfilerClockSync::evict_ack_line() const {
#if defined(__x86_64__) || defined(__i386__)
    // On the fallback path the device's PCIe writes may be non-snooped, so a cached line would read stale. The flush
    // cannot be hoisted out of the poll loop: each load re-caches the line. clflushopt was measured on Wormhole and is
    // indistinguishable from clflush here -- its advantage is pipelining a run of flushes, and this is one line
    // flushed immediately before a load that needs it back, so the cost is the re-fetch, not the ordering.
    if (hugepage_fallback_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(ack_host_ptr_)));
        _mm_lfence();
    }
#endif
}

uint32_t RealtimeProfilerClockSync::read_ack() const {
    evict_ack_line();
    return ack_host_ptr_[kSyncAckTokenWord];
}

uint64_t RealtimeProfilerClockSync::read_device_time() const {
    evict_ack_line();
    return (static_cast<uint64_t>(ack_host_ptr_[1]) << 32) | static_cast<uint64_t>(ack_host_ptr_[0]);
}

std::optional<std::chrono::nanoseconds> RealtimeProfilerClockSync::measure_rtt(
    std::chrono::steady_clock::time_point host_before, uint32_t token, std::chrono::nanoseconds timeout) {
    const auto deadline = host_before + timeout;
    uint32_t polls = 0;
    // Stop timing the moment the timestamp lands. The device sampled its clock before issuing that write, so the
    // token write and both barriers that follow only widen the interval without adding uncertainty about when the
    // clock was read. A torn read here is harmless: any change at all means the write has started.
    while (read_device_time() == last_device_time_) {
        if (++polls > kRttProbeHealthyPolls && std::chrono::steady_clock::now() > deadline) {
            return std::nullopt;  // device stopped responding
        }
    }
    const auto rtt = std::chrono::steady_clock::now() - host_before;
    const uint64_t timed_device_time = read_device_time();

    // Off the timed path: the token is ordered behind the timestamp, so it is what makes both words complete.
    while (read_ack() != token) {
        if (++polls > kRttProbeHealthyPolls && std::chrono::steady_clock::now() > deadline) {
            last_device_time_ = read_device_time();
            return std::nullopt;
        }
    }
    // The token identifies this probe's reply, and the device writes it after the timestamp, so by now device_time
    // holds this probe's value. If it moved since the read that stopped the clock, the change that stopped it came
    // from an earlier probe whose reply was still in flight, and the interval is shorter than the true round trip.
    // Such a sample re-anchors ahead of every honest one, since a smaller round trip always wins.
    if (read_device_time() != timed_device_time) {
        last_device_time_ = read_device_time();
        return std::nullopt;
    }
    return rtt;
}

std::optional<ClockSyncSample> RealtimeProfilerClockSync::probe(std::chrono::nanoseconds timeout) {
    // Opened before host_before is taken, so the zone's own cost stays outside the interval measure_rtt times.
    // Nothing between that read and the round trip's end may be instrumented for the same reason.
    TTZoneScopedDN(RT_PROFILER, "Probe");
    const auto host_before = std::chrono::steady_clock::now();
    if (++sync_seq_ == 0) {
        sync_seq_ = 1;
    }
    write_token(sync_seq_);
    const auto rtt = measure_rtt(host_before, sync_seq_, timeout);
    if (!rtt.has_value()) {
        return std::nullopt;
    }
    TTZoneValueD(RT_PROFILER, static_cast<uint64_t>(rtt->count()));
    last_device_time_ = read_device_time();
    return ClockSyncSample{host_before, *rtt, last_device_time_};
}

bool RealtimeProfilerClockSync::calibrate() {
    TTZoneScopedDN(RT_PROFILER, "Calibrate");
    // Enough that the fitted slope is dominated by the baseline rather than per-probe noise. At 5ms spacing this is
    // ~0.5s of bring-up per device.
    constexpr uint32_t kFitSamples = 100;
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr uint32_t kRunSyncMaxConsecutiveTimeouts = 3;
    // A slow handshake places its sample badly rather than merely late, so spending one of the kFitSamples slots on
    // it costs a real sample. Probe again instead, up to this many attempts, which is what keeps a degraded link from
    // stretching bring-up: worst case is this many intervals rather than kFitSamples of them.
    constexpr uint32_t kMaxProbeAttempts = kFitSamples * 3 / 2;
    // Bar for accepting a probe, against the tightest round trip seen so far on this device. Same idea as resync's
    // burst, which keeps probing until one matches the standing anchor.
    constexpr int kProbeRttOutlierFactor = 2;
    const auto host_start_time = std::chrono::steady_clock::now();

    std::vector<ClockSyncSample> samples;
    uint32_t rejected_probes = 0;
    if (ack_host_ptr_ == nullptr) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} has no host ACK word; skipping sync (using default frequency)",
            chip_id_);
    } else {
        std::this_thread::sleep_for(kRunSyncSettleDelay);

        samples.reserve(kFitSamples);
        uint32_t consecutive_timeouts = 0;
        auto best_rtt = std::chrono::nanoseconds::max();
        for (uint32_t attempt = 0; attempt < kMaxProbeAttempts && samples.size() < kFitSamples; attempt++) {
            std::this_thread::sleep_for(kRunSyncSampleInterval);

            const auto p = probe(kCalibrateProbeTimeout);
            if (!p.has_value()) {
                if (++consecutive_timeouts >= kRunSyncMaxConsecutiveTimeouts) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {} sync aborted after {} consecutive probe timeouts; the "
                        "profiler kernel may not be responding (check DPRINT output)",
                        chip_id_,
                        consecutive_timeouts);
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            // The first probe pays the cold PCIe path, so it sets no precedent for what a good round trip looks like.
            if (attempt == 0) {
                continue;
            }
            best_rtt = std::min(best_rtt, p->rtt);
            if (p->rtt > best_rtt * kProbeRttOutlierFactor) {
                ++rejected_probes;
                continue;
            }
            samples.push_back(*p);
        }
    }

    // configure() already seeded the commanded AICLK, which the model keeps if the fit has too few samples.
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

    // Measured against kFitSamples rather than what was collected, so this catches both a link that exhausted the
    // probe budget without filling the batch and one whose samples were scattered enough for the model to cut them.
    // Half still fits a slope well -- the span is unchanged and the count only enters under a square root -- but
    // below that another pass is likely to beat this one. Returning false re-runs the whole calibration through
    // calibrate_device's existing retry budget, which is what bounds the cost of a chip that keeps failing.
    if (residual->num_samples_fitted * 2 < kFitSamples) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} fit only {} of {} wanted sync samples ({} probes re-taken); retrying "
            "rather than fitting a frequency from what is left",
            chip_id_,
            residual->num_samples_fitted,
            kFitSamples,
            rejected_probes);
        return false;
    }

    // Cached only once the fit is worth reusing: try_restore_calibration hands the cached frequency to a later
    // MeshDevice without re-fitting, so a bad one would outlive the run that produced it.
    rt_profiler_frequency_cache().put(chip_id_, model_.frequency(), std::chrono::steady_clock::now());
    log_info(
        tt::LogMetal,
        "[Real-time profiler] Device {} sync complete: fit {} of {} collected samples, {} probes re-taken for a slow "
        "round trip, frequency={:.6f} GHz, fit residual rms={:.0f} ns max={:.0f} ns",
        chip_id_,
        residual->num_samples_fitted,
        residual->num_samples_offered,
        rejected_probes,
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
        return false;  // probe failed, so fall back to a full fit
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
    if (ack_host_ptr_ == nullptr) {
        return true;
    }
    // The nested Probe zones are the burst: how deep it went before a round trip matched the standing anchor.
    TTZoneScopedDN(RT_PROFILER, "Resync");
    try {
        // Keep the tightest round trip so one slow probe cannot set the published bound, but stop as soon as one
        // matches the standing anchor: each probe costs the profiler core two NOC writes and two barriers inside the
        // loop that pushes records, so probes come out of record throughput. A degraded path never finds one and pays
        // the full depth, which is exactly when the extra sampling is worth it. Whether the best of them is taken at
        // all is the model's call, not this loop's.
        constexpr int kMaxProbes = 10;
        std::optional<ClockSyncSample> best;
        for (int i = 0; i < kMaxProbes; i++) {
            const auto sample = probe(kResyncProbeTimeout);
            if (sample.has_value() && (!best.has_value() || sample->rtt < best->rtt)) {
                best = sample;
            }
            if (best.has_value() && best->rtt <= resync_exit_threshold(model_.anchor_rtt())) {
                break;
            }
        }
        if (!best.has_value()) {
            return false;
        }
        if (model_.try_reanchor(*best)) {
            publish_mapping();
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[Real-time profiler] Resync failed for device {}: {}", chip_id_, e.what());
        return false;
    }
    return true;
}

}  // namespace tt::tt_metal
