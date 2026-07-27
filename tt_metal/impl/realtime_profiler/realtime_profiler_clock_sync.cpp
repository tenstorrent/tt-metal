// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"

#include <chrono>
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
#include "hostdev/realtime_profiler_msgs.h"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {

// Round-trip busy-poll backstop. The first kRttProbeHealthyPolls reads skip the deadline check so a healthy handshake
// never reads the clock inside the round trip it is timing; only a stalled device reaches the check.
constexpr auto kRttProbeTimeout = std::chrono::microseconds(300);
constexpr uint32_t kRttProbeHealthyPolls = 128;

// How long a cached calibration stays usable across a MeshDevice close/reopen.
constexpr auto kCalibrationCacheMaxAge = std::chrono::seconds(60);

// Host ACK buffer, 32-bit words: [device_time_lo, device_time_hi, token]. device_time is at the base so it is 8-byte
// aligned; NOC PCIe writes require src/dst to share the low 4 bits, so its L1 source is 16-aligned and the token's is
// 8-mod-16.
constexpr uint32_t kSyncAckWords = 3;
constexpr uint32_t kSyncAckTokenWord = 2;

}  // namespace

std::optional<double> RealtimeProfilerFrequencyCache::try_get(
    uint32_t chip_id, std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration max_age) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = by_chip_.find(chip_id);
    if (it != by_chip_.end() && now - it->second.updated_at < max_age) {
        return it->second.frequency;
    }
    return std::nullopt;
}

void RealtimeProfilerFrequencyCache::put(
    uint32_t chip_id, double frequency, std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    by_chip_[chip_id] = Entry{frequency, now};
}

RealtimeProfilerFrequencyCache& rt_profiler_frequency_cache() {
    static RealtimeProfilerFrequencyCache cache;
    return cache;
}

void RealtimeProfilerClockSync::configure(const RealtimeProfilerClockSyncConfig& config) {
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
}

RealtimeProfilerClockSync::SyncL1Addrs RealtimeProfilerClockSync::resolve_l1_addrs(uint32_t msg_base_addr) const {
    using Msg = realtime_profiler_msgs::realtime_profiler_msg_t;
    const auto& factory =
        MetalContext::instance(context_id_).hal().get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    const auto field_addr = [&](Msg::Field field) -> uint32_t {
        return msg_base_addr + static_cast<uint32_t>(factory.offset_of<Msg>(field));
    };
    return SyncL1Addrs{
        .host_timestamp = field_addr(Msg::Field::sync_host_timestamp),
        .ack_host_addr = field_addr(Msg::Field::sync_ack_host_addr),
    };
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

void RealtimeProfilerClockSync::write_timestamp(uint32_t value) {
    // TODO: measure on Wormhole. The non-TLB path resolves the virtual core on every probe, inside the interval the
    // round trip is timed over; caching the tt_cxy_pair in configure() would take it off that path. Blackhole always
    // takes the TLB branch, so the change is unmeasurable here.
    if (sync_tlb_ != nullptr) {
        sync_tlb_->write32(l1_.host_timestamp, value);
        tt_driver_atomics::sfence();
    } else {
        const CoreCoord vcore = device_->virtual_core_from_logical_core(profiler_core_, CoreType::WORKER);
        MetalContext::instance(context_id_)
            .get_cluster()
            .write_core_immediate(&value, sizeof(value), tt_cxy_pair(device_->id(), vcore), l1_.host_timestamp);
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
    // TODO: measure on Wormhole. The flush cannot be hoisted out of the poll loop -- each load re-caches the line --
    // but clflushopt is the weakly ordered variant meant for exactly this, and would need a feature check and the
    // matching fence. Only the no-IOMMU path pays this; Blackhole polls a pinned mapping with a plain load.
    if (hugepage_fallback_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(ack_host_ptr_)));
        _mm_lfence();
    }
#endif
    return (static_cast<uint64_t>(ack_host_ptr_[1]) << 32) | static_cast<uint64_t>(ack_host_ptr_[0]);
}

std::optional<std::chrono::nanoseconds> RealtimeProfilerClockSync::measure_rtt(
    std::chrono::steady_clock::time_point host_before, uint32_t host_time_id) {
    const auto deadline = host_before + kRttProbeTimeout;
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

    // Off the timed path: the token is ordered behind the timestamp, so it is what makes both words complete.
    while (read_ack() != host_time_id) {
        if (++polls > kRttProbeHealthyPolls && std::chrono::steady_clock::now() > deadline) {
            return std::nullopt;
        }
    }
    return rtt;
}

std::optional<ClockSyncSample> RealtimeProfilerClockSync::probe() {
    const auto host_before = std::chrono::steady_clock::now();
    if (++sync_seq_ == 0) {
        sync_seq_ = 1;
    }
    write_timestamp(sync_seq_);
    const auto rtt = measure_rtt(host_before, sync_seq_);
    if (!rtt.has_value()) {
        return std::nullopt;
    }
    last_device_time_ = read_device_time();
    return ClockSyncSample{host_before, *rtt, last_device_time_};
}

bool RealtimeProfilerClockSync::run_fit() {
    // Enough that the fitted slope is dominated by the baseline rather than per-probe noise. At 5ms spacing this is
    // ~0.5s of bring-up per device.
    constexpr uint32_t kFitSamples = 100;
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr uint32_t kRunSyncMaxConsecutiveTimeouts = 3;
    const auto host_start_time = std::chrono::steady_clock::now();

    std::vector<ClockSyncSample> samples;
    if (ack_host_ptr_ == nullptr) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} has no host ACK word; skipping sync (using default frequency)",
            chip_id_);
    } else {
        std::this_thread::sleep_for(kRunSyncSettleDelay);

        uint32_t consecutive_timeouts = 0;
        for (uint32_t i = 0; i < kFitSamples + 1; i++) {
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
                    kFitSamples,
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

    // configure() already seeded the commanded AICLK, which the model keeps if the fit has too few samples.
    const ClockFitQuality quality = model_.fit(samples, host_start_time);
    if (quality.ok) {
        rt_profiler_frequency_cache().put(chip_id_, model_.frequency(), std::chrono::steady_clock::now());
        log_info(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "fit residual rms={:.0f} ns max={:.0f} ns",
            chip_id_,
            quality.num_samples,
            model_.frequency(),
            quality.residual_rms_ns,
            quality.residual_max_ns);
    } else {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using the commanded AICLK",
            chip_id_);
    }
    return quality.ok;
}

bool RealtimeProfilerClockSync::try_restore_calibration(std::chrono::steady_clock::time_point now) {
    const auto frequency = rt_profiler_frequency_cache().try_get(chip_id_, now, kCalibrationCacheMaxAge);
    if (!frequency.has_value()) {
        return false;
    }
    model_.seed_frequency(*frequency);
    resync(now);
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

bool RealtimeProfilerClockSync::resync(std::chrono::steady_clock::time_point now) {
    if (ack_host_ptr_ == nullptr) {
        return true;
    }
    try {
        const auto sample = probe();
        if (!sample.has_value()) {
            return false;
        }
        if (model_.accept_reanchor(sample->rtt, now)) {
            model_.reanchor(now, *sample);
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[Real-time profiler] Resync failed for device {}: {}", chip_id_, e.what());
        return false;
    }
    return true;
}

}  // namespace tt::tt_metal
