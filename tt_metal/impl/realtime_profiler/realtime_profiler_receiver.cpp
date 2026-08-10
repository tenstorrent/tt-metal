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

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// One probe after each drain batch. A finished record sits behind at most (host FIFO + device ring) pages, so it sees
// at most that many / kMaxSocketPagesPerRead probes before decode; history must cover those so map_record can still
// bracket the end.
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

void RealtimeProfilerReceiver::publish_pages(
    RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch) {
    TTZoneScopedDN(RT_PROFILER, "PublishBatch");
    const size_t num_pages = pages.size() / RealtimeProfilerRuntimeSizes::page_words;
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    uint64_t inverted = 0;
    uint64_t unmappable = 0;
    batch.clear();

    for (size_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = pages.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++inverted;
            continue;
        }
        const auto mapping = dev_state.clock_sync->map_record(start_timestamp, end_timestamp);
        if (!mapping.has_value()) {
            ++unmappable;
            continue;
        }
        batch.push_back(ProgramRealtimeRecord{
            .runtime_id = rp[2],
            .chip_id = dev_state.chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = mapping->frequency,
            .clock_sync =
                {.device_cycle_offset = mapping->device_cycle_offset,
                 .error = mapping->error,
                 .probe_error = mapping->probe_error,
                 .nonlinearity = mapping->nonlinearity},
            .kernel_sources = data_collector_->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])),
        });
        sync_error_window_max_ =
            sync_error_window_max_.has_value() ? std::max(*sync_error_window_max_, mapping->error) : mapping->error;
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
    if (batch.empty()) {
        return;
    }
    num_published_records_.fetch_add(batch.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_.writer().publish_batch(std::span<const ProgramRealtimeRecord>(batch));
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
        dev_state.socket->read(page_buf.data(), num_pages_to_read);
    }

    dev_state.clock_sync->resync();

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
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const uint32_t num_pages = drain_all_devices(now, page_buf);
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

uint32_t RealtimeProfilerReceiver::drain_all_devices(
    std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf) {
    uint32_t num_pages = 0;
    for (auto& dev_state : devices_) {
        try {
            const uint32_t drained = drain_device_pages(dev_state, page_buf);
            num_pages += drained;
            if (drained != 0) {
                now = std::chrono::steady_clock::now();
            } else if (dev_state.clock_sync->due_for_probe(now)) {
                TTZoneScopedDN(RT_PROFILER, "IdleSyncProbe");
                TTZoneValueD(RT_PROFILER, dev_state.chip_id);
                dev_state.clock_sync->resync();
                // an idle device may be idle because a program is running on it; this is what keeps a start that will
                // outlive the ring mappable.
                dev_state.peek_running_program_start();
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: exception while draining: {}",
                dev_state.chip_id,
                e.what());
        }
    }
    if (num_pages != 0) {
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
                "[Real-time profiler] Device {} still had {} page(s) unread when the shutdown drain gave up.",
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
