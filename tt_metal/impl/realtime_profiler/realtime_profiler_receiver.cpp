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

namespace {

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// The probe history must out-span anything still in flight: a device is probed once per drained batch and, when its
// FIFO is empty, once per sync interval, so a record that has ended sees at most pipeline-depth / batch-size probes
// (plus a handful racing the push) before it is decoded. This is what lets a record's end always find the pair of
// probes around it, however far its start may reach back -- whatever horizon the ring was sized for.
static_assert(
    DeviceClockMapping::kProbeHistoryCapacity >=
        8 * (RealtimeProfilerRuntimeSizes::fifo_pages + RT_PROFILER_RING_CAPACITY) / kMaxSocketPagesPerRead,
    "The probe history could lap past an undecoded record's end");

// Floor on how often a repeating fault is logged.
constexpr auto kWarnInterval = std::chrono::seconds(30);

constexpr size_t kMaxConsumerBatchPerDevice =
    1u << 15;                                      // records one callback may be handed at a time, per attached device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;  // hard ceiling on the above
constexpr size_t kRingHeadroomBatches = 4;         // batches of backlog the ring absorbs while a consumer works
constexpr size_t kMaxRingCapacity = 1u << 22;      // hard ceiling on the ring size

size_t consumer_batch_records_for(size_t num_devices) {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * num_devices);
}

}  // namespace

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

// Decodes what a read returned and publishes it, each record stamped from the secant between the two probes that
// surround it. Nothing is held back: the probe taken after the read is past every record in it, so the pair a record
// needs already exists by the time it is decoded.
bool RealtimeProfilerReceiver::publish_pages(
    RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch) {
    TTZoneScopedDN(RT_PROFILER, "PublishBatch");
    const size_t num_pages = pages.size() / RealtimeProfilerRuntimeSizes::page_words;
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    uint64_t rejected = 0;
    batch.clear();

    for (size_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = pages.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++rejected;
            continue;
        }
        // Only fails when both timestamps predate every retained probe, which a real record cannot; see map_record.
        const auto mapping = dev_state.clock_sync->map_record(start_timestamp, end_timestamp);
        if (!mapping.has_value()) {
            ++rejected;
            continue;
        }
        batch.push_back(ProgramRealtimeRecord{
            .runtime_id = rp[2],
            .chip_id = dev_state.chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = mapping->frequency,
            .clock_sync = {.device_cycle_offset = mapping->device_cycle_offset, .sync_error = mapping->sync_error},
            .kernel_sources = data_collector_->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])),
        });
    }

    if (rejected != 0) {
        num_malformed_records_.fetch_add(rejected, std::memory_order_relaxed);
        if (const auto now = std::chrono::steady_clock::now(); now - last_malformed_warn_ >= kWarnInterval) {
            last_malformed_warn_ = now;
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} dropped {} corrupt record(s) -- an end timestamp preceding its "
                "start, or timestamps predating every retained clock probe; these were not delivered to consumers "
                "({} in total)",
                dev_state.chip_id,
                rejected,
                num_malformed_records_.load(std::memory_order_relaxed));
        }
    }
    if (batch.empty()) {
        return false;
    }
    num_published_records_.fetch_add(batch.size(), std::memory_order_relaxed);
    ring_.writer().publish_batch(std::span<const ProgramRealtimeRecord>(batch));
    return true;
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
    register_builtin_realtime_profiler_consumers();
    return std::unique_ptr<RealtimeProfilerReceiver>(new RealtimeProfilerReceiver(context_id, std::move(devices)));
}

RealtimeProfilerReceiver::RealtimeProfilerReceiver(ContextId context_id, std::vector<RealtimeProfilerDevice> devices) :
    context_id_(context_id),
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    realtime_profiler_service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    ring_(std::min(kMaxRingCapacity, consumer_batch_records_for(devices_.size()) * kRingHeadroomBatches)) {
    // Serial: a warm-up is a few probes a sync interval apart, so a whole mesh costs milliseconds. It replaced a
    // half-second per-device fit, which is what the concurrency here existed for.
    for (RealtimeProfilerDevice& dev_state : devices_) {
        dev_state.clock_sync->warm_up();
    }
    // Resized before being cleared, not just reserved: the pages have to be touched here, because a first-touch fault
    // on the drain thread waits on mmap_lock the same way an allocation does.
    publish_batch_.resize(kMaxSocketPagesPerRead);
    publish_batch_.clear();
    realtime_profiler_service_->attach_producer(*this);

    try {
        receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run, this);
    } catch (...) {
        realtime_profiler_service_->detach_producer(*this);
        throw;
    }
}

size_t RealtimeProfilerReceiver::max_batch_records() const { return consumer_batch_records_for(devices_.size()); }

RealtimeProfilerRecordRing::Reader RealtimeProfilerReceiver::make_reader() { return ring_.make_reader(); }

void RealtimeProfilerReceiver::wait_until_no_readers() { ring_.wait_until_no_readers(); }

// Three high-water marks over the same samples, because each has one reader on its own cadence and each clears what
// it reads: all-time for "did this ever back up", one for the Tracy plot's 50ms samples, one for whatever periodic
// report is watching. Sharing an accumulator between two of them made a reader's number mean the other's window.
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

RealtimeProfilerReceiver::DrainResult RealtimeProfilerReceiver::drain_device_pages(
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
        return {};
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    {
        TTZoneScopedDN(RT_PROFILER, "SocketRead");
        TTZoneValueD(RT_PROFILER, num_pages_to_read);
        dev_state.socket->read(page_buf.data(), num_pages_to_read);
    }

    // The probe goes here, after the read and before anything is placed: every record this read returned was pushed to
    // the FIFO before the read, so it completed before the read, so this probe is past all of them, and the previous
    // one is before them. That is a bracketing pair for the whole batch obtained without waiting for anything, which is
    // the reason this path has no staging buffer, no publication gate and no deadline in it.
    {
        TTZoneScopedDN(RT_PROFILER, "ProbeAfterRead");
        TTZoneValueD(RT_PROFILER, dev_state.chip_id);
        dev_state.clock_sync->resync();
    }

    const bool published = publish_pages(
        dev_state,
        std::span(page_buf).first(num_pages_to_read * RealtimeProfilerRuntimeSizes::page_words),
        publish_batch_);
    return {.pages = num_pages_to_read, .published = published};
}

uint64_t RealtimeProfilerReceiver::run_loop(std::vector<uint32_t>& page_buf) {
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;

    constexpr auto kFifoPlotInterval = std::chrono::milliseconds(50);
    auto last_fifo_plot = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const uint32_t num_pages = drain_all_devices(now, page_buf);
        num_pages_received += num_pages;

        if (now - last_fifo_plot >= kFifoPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(fifo_pages_window_max_));
            fifo_pages_window_max_ = 0;  // cleared by its only reader, the plot above
            std::chrono::nanoseconds worst_sync_error{};
            for (const auto& dev_state : devices_) {
                worst_sync_error = std::max(worst_sync_error, dev_state.clock_sync->last_published_sync_error());
            }
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler sync error (us)",
                (std::chrono::duration<double, std::micro>{worst_sync_error}.count()));
            last_fifo_plot = now;
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

uint32_t RealtimeProfilerReceiver::drain_all_devices(
    std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf) {
    uint32_t num_pages = 0;
    bool published = false;
    for (auto& dev_state : devices_) {
        try {
            const DrainResult drained = drain_device_pages(dev_state, page_buf);
            num_pages += drained.pages;
            published |= drained.published;
            if (drained.pages != 0) {
                // Only re-read the clock after a drain that actually moved pages: an idle pass is too fast for the
                // re-read to be worth its cost.
                now = std::chrono::steady_clock::now();
            } else if (dev_state.clock_sync->due_for_probe(now)) {
                TTZoneScopedDN(RT_PROFILER, "ProbeFloor");
                TTZoneValueD(RT_PROFILER, dev_state.chip_id);
                dev_state.clock_sync->resync();
                // An idle device may be idle because a program is running on it; this is what keeps a start that will
                // outlive the ring mappable.
                dev_state.clock_sync->peek_running_program_start();
            }
        } catch (const std::exception& e) {
            if (const auto warn_now = std::chrono::steady_clock::now();
                warn_now - last_exception_warn_ >= kWarnInterval) {
                last_exception_warn_ = warn_now;
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: exception while draining: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
    }
    if (published) {
        realtime_profiler_service_->wake_consumers();
    }
    return num_pages;
}

uint64_t RealtimeProfilerReceiver::drain_on_shutdown(std::vector<uint32_t>& page_buf) {
    constexpr uint32_t kShutdownDrainQuietRounds = 10;
    constexpr auto kShutdownDrainQuietBackoff = std::chrono::milliseconds(1);
    // The socket's own teardown barrier waits for the host to have acknowledged everything the device sent, so leaving
    // this loop with pages outstanding turns into a barrier timeout in the D2H socket destructor rather than anything
    // reported here. Bounded so a device that never goes quiet cannot hang teardown either.
    constexpr auto kShutdownDrainDeadline = std::chrono::seconds(5);
    const auto give_up_at = std::chrono::steady_clock::now() + kShutdownDrainDeadline;

    uint64_t num_pages_drained = 0;
    uint32_t quiet_rounds = 0;
    while (quiet_rounds < kShutdownDrainQuietRounds && std::chrono::steady_clock::now() < give_up_at) {
        const uint32_t num_pages = drain_all_devices(std::chrono::steady_clock::now(), page_buf);
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
                "[Real-time profiler] Device {} still had {} page(s) unread when the shutdown drain gave up; the D2H "
                "socket's teardown barrier will wait for them",
                dev_state.chip_id,
                left);
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

    // One buffer pair for the thread's whole life; the steady-state loop and the shutdown drain run in sequence.
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * RealtimeProfilerRuntimeSizes::page_words);

    const uint64_t num_pages_received = run_loop(page_buf);

    // The push kernel delivers its last pages after seeing this, which the shutdown drain below collects once traffic
    // goes quiet.
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

    // detach_producer is idempotent, so a second shutdown() is harmless.
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
