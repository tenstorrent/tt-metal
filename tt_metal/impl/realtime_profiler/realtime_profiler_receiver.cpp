// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_metal.hpp>

#include <common/TracySystem.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// The ack is a posted-class store through the per-worker-core static TLB window — the
// receiver's only chip MMIO, ~0.3 us each. Batching trades apparent FIFO headroom (the device
// sees up to this many consumed-but-unacked pages) against ack traffic; 512 keeps the headroom
// loss at 1.6% for well under 1% of receiver time at the stress ceiling.
constexpr uint32_t kAckBatchPages = 512;

// Probes accrue on the kDeviceClockSyncInterval cadence, so the retained history spans
// capacity * interval (~2.5 s) of wall time. A finished record sits behind at most
// (host FIFO + device ring) pages of drain backlog — tens of milliseconds at any plausible
// production rate — so its chords are still retained when it is decoded.
static_assert(
    DeviceClockSync::kProbeHistoryCapacity >=
        (RealtimeProfilerRuntimeSizes::fifo_pages + RT_PROFILER_RING_CAPACITY) / kMaxSocketPagesPerRead,
    "The probe history could lap past an undecoded record's end");

// Floor on how often a repeating fault is logged.
constexpr auto kWarnInterval = std::chrono::seconds(30);

constexpr size_t kMaxConsumerBatchPerDevice =
    1u << 15;                                      // records one callback may be handed at a time, per attached device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;  // hard ceiling on the above
constexpr size_t kRingHeadroomBatches = 4;         // batches of backlog the ring absorbs while a consumer works
constexpr size_t kMaxRingCapacity = 1u << 22;      // hard ceiling on the ring size

uint32_t RealtimeProfilerReceiver::host_fifo_capacity_pages() const { return RealtimeProfilerRuntimeSizes::fifo_pages; }

uint32_t RealtimeProfilerReceiver::read_ring_full_wait_count() {
    uint32_t peak = 0;
    for (const auto& dev_state : devices_) {
        const uint32_t addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, ring_full_wait_count);
        std::vector<uint32_t> value(1, 0);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dev_state.device, dev_state.realtime_profiler_core, addr, sizeof(uint32_t), value, CoreType::WORKER);
        peak = std::max(peak, value[0]);
    }
    return peak;
}

size_t RealtimeProfilerReceiver::publish_pages(
    RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch) {
    TTZoneScopedDN(RT_PROFILER, "PublishBatch");
    const size_t num_pages = pages.size() / RealtimeProfilerRuntimeSizes::page_words;
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    uint64_t inverted = 0;
    uint64_t unmappable = 0;
    batch.clear();

    const uint64_t finalized = dev_state.clock_sync->finalized_device_timestamp();
    int64_t batch_max_error_ns = -1;
    const auto map_into_batch = [&](uint32_t runtime_id, uint64_t start_timestamp, uint64_t end_timestamp) {
        const auto mapping = dev_state.clock_sync->map_record(start_timestamp, end_timestamp);
        if (!mapping.has_value()) {
            ++unmappable;
            return;
        }
        batch.push_back(ProgramRealtimeRecord{
            .runtime_id = runtime_id,
            .chip_id = dev_state.chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = mapping->frequency,
            .clock_sync = {.device_cycle_offset = mapping->device_cycle_offset, .error = mapping->error},
            .kernel_sources = data_collector_->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(runtime_id)),
        });
        batch_max_error_ns = std::max(batch_max_error_ns, mapping->error.count());
    };

    // Held-back records first: they predate anything decoded below.
    auto pending_split = dev_state.pending_records.begin();
    for (; pending_split != dev_state.pending_records.end() && pending_split->end_timestamp <= finalized;
         ++pending_split) {
        map_into_batch(pending_split->runtime_id, pending_split->start_timestamp, pending_split->end_timestamp);
    }
    dev_state.pending_records.erase(dev_state.pending_records.begin(), pending_split);

    for (size_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = pages.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++inverted;
            continue;
        }
        if (end_timestamp > finalized) {
            TT_ASSERT(dev_state.pending_records.size() < dev_state.pending_records.capacity());
            dev_state.pending_records.push_back(PendingRealtimeRecord{
                .start_timestamp = start_timestamp, .end_timestamp = end_timestamp, .runtime_id = rp[2]});
            continue;
        }
        map_into_batch(rp[2], start_timestamp, end_timestamp);
    }

    if (inverted != 0) {
        const uint64_t total =
            num_inverted_timestamp_records_.fetch_add(inverted, std::memory_order_relaxed) + inverted;
        if (const auto now = std::chrono::steady_clock::now(); now - last_inverted_timestamp_warn_ >= kWarnInterval) {
            last_inverted_timestamp_warn_ = now;
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} dropped {} record(s) with end_timestamp < start_timestamp ({} in "
                "total)",
                dev_state.chip_id,
                inverted,
                total);
        }
    }
    if (unmappable != 0) {
        const uint64_t total = num_unmappable_records_.fetch_add(unmappable, std::memory_order_relaxed) + unmappable;
        if (const auto now = std::chrono::steady_clock::now(); now - last_unmappable_warn_ >= kWarnInterval) {
            last_unmappable_warn_ = now;
            log_error(
                tt::LogMetal,
                "[Real-time profiler] Device {} dropped {} unmappable record(s): no retained clock probe precedes "
                "their timestamps ({} in total).",
                dev_state.chip_id,
                unmappable,
                total);
        }
    }
    if (batch_max_error_ns >= 0) {
        const std::chrono::nanoseconds batch_max{batch_max_error_ns};
        sync_error_window_max_ =
            sync_error_window_max_.has_value() ? std::max(*sync_error_window_max_, batch_max) : batch_max;
    }
    if (batch.empty()) {
        return 0;
    }
    num_published_records_.fetch_add(batch.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_.writer().publish_batch(std::span<const ProgramRealtimeRecord>(batch));
    return batch.size();
}

std::unique_ptr<RealtimeProfilerReceiver> RealtimeProfilerReceiver::create(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const ContextId context_id = mesh_device->impl().get_context_id();
    auto devices = initialize_realtime_profiler_devices(mesh_device, context_id);
    if (devices.empty()) {
        log_debug(
            tt::LogMetal, "[Real-time profiler] No local devices found in mesh, skipping real-time profiler setup");
        return nullptr;
    }
    return std::unique_ptr<RealtimeProfilerReceiver>(new RealtimeProfilerReceiver(context_id, std::move(devices)));
}

RealtimeProfilerReceiver::RealtimeProfilerReceiver(ContextId context_id, std::vector<RealtimeProfilerDevice> devices) :
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    realtime_profiler_service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    ring_(std::min(kMaxRingCapacity, max_batch_records() * kRingHeadroomBatches)) {
    // A batch can carry a full drain of fresh records plus a full drain of held-back ones.
    publish_batch_.resize(2 * kMaxSocketPagesPerRead);
    publish_batch_.clear();
    for (RealtimeProfilerDevice& dev_state : devices_) {
        dev_state.pending_records.reserve(kMaxSocketPagesPerRead);
    }
    // Warm-up here, not at device construction: any earlier leaves a multi-hundred-ms probe gap
    // while the remaining devices initialize, and startup records pick up fallback-tier bounds.
    constexpr int kWarmUpRounds = 4;
    for (int round = 0; round < kWarmUpRounds; ++round) {
        if (round != 0) {
            std::this_thread::sleep_for(kDeviceClockSyncInterval);
        }
        for (RealtimeProfilerDevice& dev_state : devices_) {
            if (dev_state.clock_sync->has_direct_clock_read()) {
                dev_state.clock_sync->ingest_probe(dev_state.clock_sync->read_probe());
            }
        }
    }
    // Grouped phases: probing every device at the same instant advances every holdback
    // watermark together, and one drain pass then publishes the whole mesh's held records — a
    // synchronized burst that sets the FIFO high-water mark. Per-device cadence is unaffected.
    constexpr size_t kProbePhaseGroups = 4;
    probe_next_due_.resize(devices_.size());
    const auto schedule_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < devices_.size(); ++i) {
        probe_next_due_[i] = schedule_start + (i % kProbePhaseGroups) * (kDeviceClockSyncInterval / kProbePhaseGroups);
    }
    realtime_profiler_service_->attach_producer(*this);

    try {
        sync_thread_ = std::thread(&RealtimeProfilerReceiver::run_sync, this);
        receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run, this);
    } catch (...) {
        sync_stop_.store(true, std::memory_order_release);
        if (sync_thread_.joinable()) {
            sync_thread_.join();
        }
        realtime_profiler_service_->detach_producer(*this);
        throw;
    }
}

size_t RealtimeProfilerReceiver::max_batch_records() const {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * devices_.size());
}

RealtimeProfilerRecordRing::Reader RealtimeProfilerReceiver::make_reader() { return ring_.make_reader(); }

void RealtimeProfilerReceiver::wait_until_no_readers() { ring_.wait_until_no_readers(); }

void RealtimeProfilerReceiver::note_fifo_depth(uint32_t available) {
    // Single writer, so a load-then-store is a max.
    if (available > peak_fifo_pages_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_.store(available, std::memory_order_relaxed);
    }
    if (available > peak_fifo_pages_since_report_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_since_report_.store(available, std::memory_order_relaxed);
    }
    fifo_pages_window_max_ = std::max(fifo_pages_window_max_, available);
}

uint64_t RealtimeProfilerReceiver::take_peak_probe_gap_ns() {
    int64_t peak = 0;
    for (auto& dev_state : devices_) {
        peak = std::max(peak, dev_state.clock_sync->take_peak_probe_gap().count());
    }
    return static_cast<uint64_t>(peak);
}

uint64_t RealtimeProfilerReceiver::num_chords_finalized() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_sync->num_finalized_chords();
    }
    return total;
}

uint64_t RealtimeProfilerReceiver::num_chords_certified() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_sync->num_certified_chords();
    }
    return total;
}

uint64_t RealtimeProfilerReceiver::num_records_on_uncertified_chords() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_sync->num_records_on_uncertified_chords();
    }
    return total;
}

void RealtimeProfilerReceiver::ingest_probes(RealtimeProfilerDevice& dev_state) {
    std::array<DeviceClockSync::Anchor, 64> buf;
    for (;;) {
        const auto got = dev_state.probe_reader->read_batch(std::span<DeviceClockSync::Anchor>(buf));
        for (const DeviceClockSync::Anchor& probe : got) {
            dev_state.clock_sync->ingest_probe(probe);
        }
        if (got.size() < buf.size()) {
            return;
        }
    }
}

std::chrono::steady_clock::time_point RealtimeProfilerReceiver::probe_due_devices() {
    const auto now = std::chrono::steady_clock::now();
    auto earliest = now + kDeviceClockSyncInterval;
    for (size_t i = 0; i < devices_.size(); ++i) {
        if (now >= probe_next_due_[i]) {
            devices_[i].probe_ring->writer().publish(devices_[i].clock_sync->read_probe());
            // Absolute schedule: a late probe is followed by a correspondingly early one, so an
            // adjacent chord pair absorbs the lateness instead of compounding it — the
            // certificate budget cares about pair sums, not single gaps.
            probe_next_due_[i] = std::max(probe_next_due_[i] + kDeviceClockSyncInterval, now);
        }
        earliest = std::min(earliest, probe_next_due_[i]);
    }
    return earliest;
}

void RealtimeProfilerReceiver::run_sync() {
    tracy::SetThreadName("RtProfilerSync");
#if defined(__linux__)
    ::prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    while (!sync_stop_.load(std::memory_order_acquire)) {
        const auto earliest = probe_due_devices();
        std::this_thread::sleep_until(std::min(earliest, std::chrono::steady_clock::now() + kDeviceClockSyncInterval));
    }
}

uint32_t RealtimeProfilerReceiver::drain_device_pages(
    RealtimeProfilerDevice& dev_state, std::vector<uint32_t>& page_buf) {
    const uint32_t available = dev_state.socket->pages_available();
    note_fifo_depth(available);
    if (available >= RealtimeProfilerRuntimeSizes::fifo_pages && !dev_state.fifo_capacity_warned) {
        dev_state.fifo_capacity_warned = true;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} D2H FIFO reached capacity ({} pages); profiler data may be dropped",
            dev_state.chip_id,
            RealtimeProfilerRuntimeSizes::fifo_pages);
    }
    if (available == 0) {
        return 0;
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    {
        TTZoneScopedDN(RT_PROFILER, "SocketRead");
        TTZoneValueD(RT_PROFILER, num_pages_to_read);
        dev_state.socket->read(page_buf.data(), num_pages_to_read, /*notify_sender=*/false);
    }
    dev_state.unacked_pages += num_pages_to_read;
    if (dev_state.unacked_pages >= kAckBatchPages) {
        dev_state.socket->notify_sender();
        dev_state.unacked_pages = 0;
    }

    ingest_probes(dev_state);

    publish_pages(
        dev_state,
        std::span(page_buf).first(num_pages_to_read * RealtimeProfilerRuntimeSizes::page_words),
        publish_batch_);
    return num_pages_to_read;
}

uint64_t RealtimeProfilerReceiver::run_loop(std::vector<uint32_t>& page_buf) {
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;

    constexpr auto kPlotInterval = std::chrono::milliseconds(1);
    auto last_plot = std::chrono::steady_clock::now();
    auto last_iteration = last_plot;
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const auto iteration_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_iteration).count());
        last_iteration = now;
        if (iteration_ns > peak_loop_ns_.load(std::memory_order_relaxed)) {
            peak_loop_ns_.store(iteration_ns, std::memory_order_relaxed);
        }
        loop_count_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t num_pages = drain_all_devices(page_buf);
        num_pages_received += num_pages;

        if (now - last_plot >= kPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(fifo_pages_window_max_));
            fifo_pages_window_max_ = 0;
            if (sync_error_window_max_.has_value()) {
                [[maybe_unused]] const double sync_error_us =
                    std::chrono::duration<double, std::micro>{*sync_error_window_max_}.count();
                TracyPlot("RT profiler sync error (us)", sync_error_us);
                sync_error_window_max_.reset();
            }
            last_plot = now;
        }

        if (num_pages > 0) {
            backoff = std::chrono::microseconds{1};
            continue;
        }
        std::this_thread::sleep_for(backoff);
        backoff += std::max(backoff / 4, std::chrono::microseconds{1});
        backoff = std::min(backoff, kReceiverMaxBackoff);
    }
    return num_pages_received;
}

uint32_t RealtimeProfilerReceiver::drain_all_devices(std::vector<uint32_t>& page_buf) {
    uint32_t num_pages = 0;
    bool published_on_idle = false;
    for (auto& dev_state : devices_) {
        try {
            const uint32_t drained = drain_device_pages(dev_state, page_buf);
            num_pages += drained;
            if (drained == 0) {
                ingest_probes(dev_state);
                if (!dev_state.pending_records.empty()) {
                    published_on_idle |= publish_pages(dev_state, {}, publish_batch_) != 0;
                }
                if (dev_state.unacked_pages != 0) {
                    dev_state.socket->notify_sender();
                    dev_state.unacked_pages = 0;
                }
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: exception while draining: {}",
                dev_state.chip_id,
                e.what());
        }
    }
    if (num_pages != 0 || published_on_idle) {
        realtime_profiler_service_->wake_consumers();
    }
    return num_pages;
}

uint64_t RealtimeProfilerReceiver::drain_on_shutdown(std::vector<uint32_t>& page_buf) {
    constexpr uint32_t kShutdownDrainQuietRounds = 10;
    constexpr auto kShutdownDrainQuietBackoff = std::chrono::milliseconds(1);
    constexpr auto kShutdownDrainDeadline = std::chrono::seconds(5);
    const auto give_up_at = std::chrono::steady_clock::now() + kShutdownDrainDeadline;

    uint64_t num_pages_drained = 0;
    uint32_t quiet_rounds = 0;
    while (quiet_rounds < kShutdownDrainQuietRounds && std::chrono::steady_clock::now() < give_up_at) {
        const uint32_t num_pages = drain_all_devices(page_buf);
        num_pages_drained += num_pages;
        bool outstanding = false;
        for (const RealtimeProfilerDevice& dev_state : devices_) {
            outstanding = outstanding || dev_state.socket->pages_available() != 0;
        }
        if (num_pages != 0 || outstanding) {
            quiet_rounds = 0;
        } else {
            quiet_rounds++;
        }
        std::this_thread::sleep_for(kShutdownDrainQuietBackoff);
    }

    for (const RealtimeProfilerDevice& dev_state : devices_) {
        if (const uint32_t left = dev_state.socket->pages_available(); left != 0) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} still had {} page(s) unread when the shutdown drain gave up.",
                dev_state.chip_id,
                left);
        }
        if (!dev_state.pending_records.empty()) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} still had {} record(s) held back when the shutdown drain gave up.",
                dev_state.chip_id,
                dev_state.pending_records.size());
        }
    }
    return num_pages_drained;
}

void RealtimeProfilerReceiver::run() {
    tracy::SetThreadName("RealtimeProfiler");
#if defined(__linux__)
    ::prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * RealtimeProfilerRuntimeSizes::page_words);

    const uint64_t num_pages_received = run_loop(page_buf);

    for (auto& dev_state : devices_) {
        const uint32_t terminate_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, terminate);
        std::vector<uint32_t> terminate_flag = {1};
        try {
            tt::tt_metal::detail::WriteToDeviceL1(
                dev_state.device, dev_state.realtime_profiler_core, terminate_addr, terminate_flag, CoreType::WORKER);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Failed to write terminate flag for device {}: {}",
                dev_state.chip_id,
                e.what());
        }
    }

    const uint64_t num_pages_drained = drain_on_shutdown(page_buf);

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown)",
        num_pages_received + num_pages_drained,
        num_pages_drained);
}

RealtimeProfilerReceiver::~RealtimeProfilerReceiver() { shutdown(); }

void RealtimeProfilerReceiver::shutdown() {
    if (receiver_thread_.joinable()) {
        stop_.store(true, std::memory_order_release);
        receiver_thread_.join();
    }
    if (sync_thread_.joinable()) {
        sync_stop_.store(true, std::memory_order_release);
        sync_thread_.join();
    }

    realtime_profiler_service_->detach_producer(*this);

    for (const auto& dev_state : devices_) {
        const uint32_t full_wait_addr =
            dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, ring_full_wait_count);
        std::vector<uint32_t> full_wait(1, 0);
        try {
            tt::tt_metal::detail::ReadFromDeviceL1(
                dev_state.device,
                dev_state.realtime_profiler_core,
                full_wait_addr,
                sizeof(uint32_t),
                full_wait,
                CoreType::WORKER);
            if (full_wait[0] != 0) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} L1 ring hit capacity {} time(s); profiler records may have been "
                    "dropped",
                    dev_state.chip_id,
                    full_wait[0]);
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Failed to read ring_full_wait_count for device {}: {}",
                dev_state.chip_id,
                e.what());
        }
    }

    devices_.clear();
}

}  // namespace tt::tt_metal
