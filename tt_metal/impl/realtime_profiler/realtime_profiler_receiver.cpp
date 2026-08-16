// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
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

// Cap on matured held-back records published per publish_pages call. A deep holdback backlog
// (a probe outage's worth) then flushes across several passes instead of one unbounded pass
// that would starve every other device's drain and acks; the worst full pass stays bounded by
// kMaxPendingFlushPerPass + kMaxSocketPagesPerRead records per device.
constexpr size_t kMaxPendingFlushPerPass = 4096;

// The ack is a posted-class store through the per-worker-core static TLB window — the
// receiver's only chip MMIO, ~0.3 us each. Batching trades apparent FIFO headroom (the device
// sees up to this many consumed-but-unacked pages) against ack traffic; 512 keeps the headroom
// loss at 1.6% for well under 1% of receiver time at the stress ceiling.
constexpr uint32_t kAckBatchPages = 512;

// Floor on how often a repeating fault is logged.
constexpr auto kWarnInterval = std::chrono::seconds(30);

constexpr size_t kMaxConsumerBatchPerDevice =
    1u << 15;                                      // records one callback may be handed at a time, per attached device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;  // hard ceiling on the above
constexpr size_t kRingHeadroomBatches = 4;         // batches of backlog the ring absorbs while a consumer works
constexpr size_t kMaxRingCapacity = 1u << 22;      // hard ceiling on the ring size

uint32_t RealtimeProfilerReceiver::host_fifo_capacity_pages() const { return RealtimeProfilerRuntimeSizes::fifo_pages; }

namespace {
uint32_t read_device_ring_full_wait(const RealtimeProfilerDevice& dev_state) {
    const uint32_t addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, ring_full_wait_count);
    std::vector<uint32_t> value(1, 0);
    tt::tt_metal::detail::ReadFromDeviceL1(
        dev_state.device, dev_state.realtime_profiler_core, addr, sizeof(uint32_t), value, CoreType::WORKER);
    return value[0];
}
}  // namespace

uint32_t RealtimeProfilerReceiver::read_ring_full_wait_count() {
    uint32_t peak = 0;
    for (const auto& dev_state : devices_) {
        peak = std::max(peak, read_device_ring_full_wait(dev_state));
    }
    return peak;
}

size_t RealtimeProfilerReceiver::publish_pages(
    RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch) {
    TTZoneScopedDN(RT_PROFILER, "PublishBatch");
    const size_t num_pages = pages.size() / RealtimeProfilerRuntimeSizes::page_words;
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    uint64_t inverted = 0;
    batch.clear();

    const uint64_t finalized = dev_state.clock_view->finalized_device_timestamp();
    int64_t batch_max_error_ns = -1;
    const auto map_into_batch = [&](uint32_t runtime_id, uint64_t start_timestamp, uint64_t end_timestamp) {
        const auto mapping = dev_state.clock_view->map_record(start_timestamp, end_timestamp);
        batch.push_back(ProgramRealtimeRecord{
            .runtime_id = runtime_id,
            .chip_id = dev_state.chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = mapping.frequency,
            .clock_sync = {.device_cycle_offset = mapping.device_cycle_offset, .error = mapping.error},
            .kernel_sources = data_collector_->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(runtime_id)),
        });
        batch_max_error_ns = std::max(batch_max_error_ns, mapping.error.count());
    };

    // Held-back records first: they predate anything decoded below.
    size_t flushed = 0;
    while (!dev_state.pending_records.empty() && flushed < kMaxPendingFlushPerPass &&
           dev_state.pending_records.front().end_timestamp <= finalized) {
        const PendingRealtimeRecord r = dev_state.pending_records.pop_front();
        map_into_batch(r.runtime_id, r.start_timestamp, r.end_timestamp);
        ++flushed;
    }

    uint64_t evicted = 0;
    for (size_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = pages.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++inverted;
            continue;
        }
        // A fresh matured record may bypass the ring only when the ring is empty: after a capped
        // flush the ring can still hold matured records older than this one, and publication must
        // stay oldest-first (the batch contract).
        if (end_timestamp > finalized || !dev_state.pending_records.empty()) {
            if (dev_state.pending_records.full()) {
                const PendingRealtimeRecord oldest = dev_state.pending_records.pop_front();
                map_into_batch(oldest.runtime_id, oldest.start_timestamp, oldest.end_timestamp);
                ++evicted;
            }
            dev_state.pending_records.push_back(PendingRealtimeRecord{
                .start_timestamp = start_timestamp, .end_timestamp = end_timestamp, .runtime_id = rp[2]});
            continue;
        }
        map_into_batch(rp[2], start_timestamp, end_timestamp);
    }
    if (evicted != 0) {
        const uint64_t total = num_holdback_evictions_.fetch_add(evicted, std::memory_order_relaxed) + evicted;
        if (const auto now = std::chrono::steady_clock::now(); now - last_eviction_warn_ >= kWarnInterval) {
            last_eviction_warn_ = now;
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} published {} held-back record(s) early with fallback-quality bounds; "
                "the holdback ring filled during a clock-probe outage ({} in total)",
                dev_state.chip_id,
                evicted,
                total);
        }
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
    // Declared before the scheduler: registered clock syncs must outlive it, so any unwind from
    // here on joins the sweep before the devices die.
    std::vector<RealtimeProfilerDevice> devices;
    auto probe_scheduler = std::make_unique<ProbeScheduler>(kDeviceClockSyncInterval);
    auto probe_demand = probe_scheduler->demand();
    devices = initialize_realtime_profiler_devices(mesh_device, context_id);
    if (devices.empty()) {
        log_debug(
            tt::LogMetal, "[Real-time profiler] No local device could run the real-time profiler, skipping setup");
        return nullptr;
    }
    for (RealtimeProfilerDevice& dev_state : devices) {
        probe_scheduler->register_device(*dev_state.clock_sync);
    }
    return std::unique_ptr<RealtimeProfilerReceiver>(new RealtimeProfilerReceiver(
        context_id, std::move(devices), std::move(probe_scheduler), std::move(probe_demand)));
}

RealtimeProfilerReceiver::RealtimeProfilerReceiver(
    ContextId context_id,
    std::vector<RealtimeProfilerDevice> devices,
    std::unique_ptr<ProbeScheduler> probe_scheduler,
    ProbeScheduler::Demand probe_demand) :
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    realtime_profiler_service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    probe_scheduler_(std::move(probe_scheduler)),
    probe_demand_(std::move(probe_demand)),
    ring_(std::min(kMaxRingCapacity, max_batch_records() * kRingHeadroomBatches)) {
    // Structural batch bound: a capped pending flush plus one drain's worth of fresh records
    // (each fresh record contributes one publish — direct, or an eviction it forced). Sized and
    // touched here so publish_pages can never reach the allocator.
    publish_batch_.resize(kMaxPendingFlushPerPass + kMaxSocketPagesPerRead);
    publish_batch_.clear();
    realtime_profiler_service_->attach_producer(ring_, max_batch_records());

    try {
        receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run, this);
    } catch (...) {
        realtime_profiler_service_->detach_producer(ring_);
        throw;
    }
}

size_t RealtimeProfilerReceiver::max_batch_records() const {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * devices_.size());
}

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
        total += dev_state.clock_view->num_finalized_chords();
    }
    return total;
}

uint64_t RealtimeProfilerReceiver::num_chords_certified() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_view->num_certified_chords();
    }
    return total;
}

uint64_t RealtimeProfilerReceiver::num_records_on_uncertified_chords() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_view->num_records_on_uncertified_chords();
    }
    return total;
}

uint64_t RealtimeProfilerReceiver::num_rejected_probes() const {
    uint64_t total = 0;
    for (const auto& dev_state : devices_) {
        total += dev_state.clock_sync->num_rejected_probes() + dev_state.clock_view->num_discarded_probes();
    }
    return total;
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

    dev_state.clock_view->ingest_queued_probes();

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

        // With no registered consumers, records are undeliverable: pause probing and discard
        // pages unread (still acked, so the device never backs up). A late consumer sees only
        // post-registration records anyway, and its first chords converge from fallback tier
        // within a few resumed sweeps.
        const bool has_consumers = realtime_profiler_service_->has_consumers();
        if (has_consumers != consumers_active_) {
            consumers_active_ = has_consumers;
            if (has_consumers) {
                probe_demand_.emplace(*probe_scheduler_);
            } else {
                probe_demand_.reset();
                for (RealtimeProfilerDevice& dev_state : devices_) {
                    dev_state.pending_records.clear();
                    if (dev_state.unacked_pages != 0) {
                        dev_state.socket->notify_sender();
                        dev_state.unacked_pages = 0;
                    }
                }
            }
        }
        if (!consumers_active_) {
            for (RealtimeProfilerDevice& dev_state : devices_) {
                dev_state.socket->discard_pending_pages();
            }
            std::this_thread::sleep_for(kReceiverMaxBackoff);
            continue;
        }

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
        const uint32_t drained = drain_device_pages(dev_state, page_buf);
        num_pages += drained;
        if (drained == 0) {
            dev_state.clock_view->ingest_queued_probes();
            if (!dev_state.pending_records.empty()) {
                published_on_idle |= publish_pages(dev_state, {}, publish_batch_) != 0;
            }
            if (dev_state.unacked_pages != 0) {
                dev_state.socket->notify_sender();
                dev_state.unacked_pages = 0;
            }
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
            outstanding = outstanding || dev_state.socket->pages_available() != 0 || !dev_state.pending_records.empty();
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

    // The drain needs the probe cadence to advance holdback watermarks even if the consumer
    // gate dropped its demand.
    const auto drain_demand = probe_scheduler_->demand();

    for (auto& dev_state : devices_) {
        const uint32_t terminate_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, terminate);
        std::vector<uint32_t> terminate_flag = {1};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device, dev_state.realtime_profiler_core, terminate_addr, terminate_flag, CoreType::WORKER);
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
    // The scheduler outlives the receiver thread so the shutdown drain can still finalize
    // held-back records; stopping it here is what ends the cadence. The demand token must not
    // outlive the scheduler it counts on.
    probe_demand_.reset();
    probe_scheduler_.reset();

    realtime_profiler_service_->detach_producer(ring_);

    for (const auto& dev_state : devices_) {
        if (const uint32_t full_wait = read_device_ring_full_wait(dev_state); full_wait != 0) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} L1 ring hit capacity {} time(s); profiler records may have been "
                "dropped",
                dev_state.chip_id,
                full_wait);
        }
    }

    devices_.clear();
}

}  // namespace tt::tt_metal
