// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_metal.hpp>

#include <common/TracySystem.hpp>
#include <llrt/tt_cluster.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_tracy_consumer.hpp"
#include "tt_metal/impl/streaming_profiler/device_wall_clock.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

// Dispatch-stall marker ID — must match device-side REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID. Page words 0-1:
// device time the stall ended, word 2: its length in device cycles.
constexpr uint32_t REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID = 0xFFFFFFFE;

// Device cycles to microseconds; frequency is in cycles per nanosecond.
double cycles_to_us(uint64_t cycles, double frequency) { return static_cast<double>(cycles) / (frequency * 1000.0); }

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;
// Pages read from a device before the host acknowledges them. Until then the device counts them as still in its
// FIFO.
constexpr uint32_t kAckBatchPages = 256;

constexpr size_t kMaxConsumerBatchPerDevice = 1u << 15;  // max batch size per device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;        // hard cap on total batch size (corresponds to 32 devices)
constexpr size_t kRingHeadroomBatches = 4;
constexpr size_t kMaxRingCapacity = 1u << 22;

// How often each device's clock mapping is re-anchored, and how many reads of its wall clock one anchor takes.
// The device clock's rate changes fastest just after load starts, so anchors come more often then.
constexpr auto kAnchorPeriod = std::chrono::seconds(1);
constexpr auto kEarlyAnchorPeriod = std::chrono::milliseconds(100);
constexpr auto kEarlyAnchorWindow = std::chrono::seconds(5);
constexpr auto kSettlingAnchorPeriod = std::chrono::milliseconds(250);
constexpr auto kSettlingAnchorWindow = std::chrono::seconds(10);

std::chrono::steady_clock::duration anchor_period(std::chrono::steady_clock::duration since_start) {
    if (since_start < kEarlyAnchorWindow) {
        return kEarlyAnchorPeriod;
    }
    return since_start < kSettlingAnchorWindow ? kSettlingAnchorPeriod : kAnchorPeriod;
}
constexpr uint32_t kAnchorReads = 4;

// The host clock readings are taken on: Tracy's when Tracy is built in, so zones need no conversion.
int64_t host_ticks() {
#ifdef TRACY_ENABLE
    return TracyGetCpuTime();
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

double ns_per_host_tick() {
#ifdef TRACY_ENABLE
    return TracyGetTimerMul();
#else
    return 1.0;
#endif
}

// Starts the device's clock mapping: its first reading, with the nominal frequency until an anchor measures it.
void init_clock_mapping(tt::Cluster& cluster, RealtimeProfilerDevice& dev_state) {
    dev_state.wall_clock = std::make_unique<DeviceWallClock>(
        cluster,
        dev_state.chip_id,
        cluster.get_virtual_coordinate_from_logical_coordinates(
            dev_state.chip_id, dev_state.realtime_profiler_core, CoreType::WORKER));
    const DeviceWallClock::Reading reading = dev_state.wall_clock->read(kAnchorReads, host_ticks);
    dev_state.anchor_device = reading.wall;
    dev_state.anchor_host = reading.host;
    dev_state.frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
}

// Takes a new reading of the device's clock and updates the mapping's offset and frequency from it.
void anchor_clock_mapping(RealtimeProfilerDevice& dev_state) {
    const DeviceWallClock::Reading reading = dev_state.wall_clock->read(kAnchorReads, host_ticks);
    const double elapsed_ns = static_cast<double>(reading.host - dev_state.anchor_host) * ns_per_host_tick();
    // A read that straddles the low word's wrap while a RISC on that core reads its own clock comes back 2^32 cycles
    // off. The mapping being replaced says where the clock should be.
    const double expected = static_cast<double>(dev_state.anchor_device) + elapsed_ns * dev_state.frequency;
    if (std::abs(static_cast<double>(reading.wall) - expected) > static_cast<double>(1ull << 31)) {
        return;
    }
    // The clock's advance since the previous anchor gives its frequency.
    dev_state.frequency = static_cast<double>(reading.wall - dev_state.anchor_device) / elapsed_ns;
    dev_state.anchor_device = reading.wall;
    dev_state.anchor_host = reading.host;
}

// Tells the device how far the host has read, which frees that space in its FIFO.
void acknowledge(RealtimeProfilerDevice& dev_state) {
    dev_state.socket->pop(0);
    dev_state.unacked_pages = 0;
}

size_t max_consumer_batch_records(size_t num_devices) {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * num_devices);
}

}  // namespace

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
    context_id_(context_id),
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    ring_(std::min(kMaxRingCapacity, max_consumer_batch_records(devices_.size()) * kRingHeadroomBatches)) {
    data_collector_->AddProgramMetadataReader();
    page_buf_.resize(kMaxSocketPagesPerRead * RealtimeProfilerRuntimeSizes::page_words);
    record_buf_.reserve(kMaxSocketPagesPerRead);

    auto& cluster = MetalContext::instance(context_id).get_cluster();
    for (auto& dev_state : devices_) {
        init_clock_mapping(cluster, dev_state);
        service_->tracy().add_device(
            dev_state.chip_id, dev_state.anchor_host, dev_state.anchor_device, dev_state.frequency);
        publish_device_profiler_sync_anchor(dev_state);
    }

    service_->attach_producer(ring_, max_consumer_batch_records(devices_.size()));

    // Background receiver thread that polls all device sockets round-robin
    receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run_receiver, this);
}

void RealtimeProfilerReceiver::publish_device_profiler_sync_anchor(const RealtimeProfilerDevice& dev_state) {
    // Accumulate-only: there the device profiler skips its own sync and borrows this mapping; otherwise it runs its
    // own sync, so realtime_sync_line is left unset.
    if (!MetalContext::instance(context_id_).rtoptions().get_profiler_accumulate()) {
        return;
    }
    // The raw anchor (host TSC, device cycle, frequency), not a SyncInfo: the device profiler keeps its own worker
    // anchor and only adopts the host-device mapping. Valid because all cores share one wall clock.
    auto& psm = MetalContext::instance(context_id_).profiler_state_manager();
    if (!psm || !psm->device_profiler_map.contains(dev_state.chip_id)) {
        return;
    }
    std::lock_guard<std::recursive_mutex> map_lock(psm->device_profiler_map_mutex);
    psm->device_profiler_map.at(dev_state.chip_id).realtime_sync_line = tt::tt_metal::DeviceProfiler::RealtimeSyncLine{
        static_cast<double>(dev_state.anchor_host), static_cast<double>(dev_state.anchor_device), dev_state.frequency};
}

uint32_t RealtimeProfilerReceiver::host_fifo_capacity_pages() const { return RealtimeProfilerRuntimeSizes::fifo_pages; }

uint32_t RealtimeProfilerReceiver::ring_full_wait_count() const {
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

uint32_t RealtimeProfilerReceiver::record_ring_full_wait_count() const {
    uint32_t peak = 0;
    for (const auto& dev_state : devices_) {
        if (!dev_state.dispatch_s_core) {
            continue;
        }
        std::vector<uint32_t> value(1, 0);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dev_state.device,
            *dev_state.dispatch_s_core,
            dev_state.record_full_wait_count_addr,
            sizeof(uint32_t),
            value,
            CoreType::WORKER);
        peak = std::max(peak, value[0]);
    }
    return peak;
}

void RealtimeProfilerReceiver::report_dispatch_stall(
    RealtimeProfilerDevice& dev_state, uint64_t stall_end_timestamp, uint32_t stall_cycles) {
    dev_state.dispatch_stall_events++;
    dev_state.dispatch_stall_cycles += stall_cycles;
    dispatch_stall_events_.fetch_add(1, std::memory_order_relaxed);
    dispatch_stall_cycles_.fetch_add(stall_cycles, std::memory_order_relaxed);
    if (dev_state.dispatch_stall_events == 1) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {}: dispatch stalled {:.2f} us waiting for the real-time profiler (record "
            "ring full); profiler records are delayed, not lost. Further stalls are summarized at shutdown.",
            dev_state.chip_id,
            cycles_to_us(stall_cycles, dev_state.frequency));
    }
    service_->tracy().push_dispatch_stall_marker(dev_state.chip_id, stall_end_timestamp, stall_cycles);
}

void RealtimeProfilerReceiver::publish_pages(RealtimeProfilerDevice& dev_state, uint32_t num_pages) {
    record_buf_.clear();
    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = page_buf_.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        if (rp[3] == REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID) {
            report_dispatch_stall(dev_state, (static_cast<uint64_t>(rp[0]) << 32) | rp[1], rp[2]);
            continue;
        }
        const uint64_t start = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end = (static_cast<uint64_t>(rp[4]) << 32) | rp[5];
        // A record that ends before it starts has no duration to report. It happens in the startup race where a
        // compute kernel sees dispatch_d clear the stream registers before dispatch_s stamps the start.
        if (end < start) {
            if (num_end_before_start_records_.fetch_add(1, std::memory_order_relaxed) == 0) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: dropped the record of program {}, which ends before it starts "
                    "(start={}, end={}). Further drops are counted and reported at close.",
                    dev_state.chip_id,
                    rp[2],
                    start,
                    end);
            }
            continue;
        }
        record_buf_.emplace_back(
            rp[2],
            dev_state.chip_id,
            start,
            end,
            dev_state.frequency,
            data_collector_->GetProgramMetadata(static_cast<uint16_t>(rp[2])).kernel_sources);
    }
    if (record_buf_.empty()) {
        return;
    }
    num_published_records_.fetch_add(record_buf_.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_.writer().publish_batch(std::span<const experimental::ProgramRealtimeRecord>(record_buf_));
}

void RealtimeProfilerReceiver::note_fifo_depth(RealtimeProfilerDevice& dev_state, uint32_t available) {
    if (available > peak_fifo_pages_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_.store(available, std::memory_order_relaxed);
    }
    windowed_peak_fifo_pages_ = std::max(windowed_peak_fifo_pages_, available);
    if (available >= RealtimeProfilerRuntimeSizes::fifo_pages && !dev_state.fifo_reached_capacity) {
        dev_state.fifo_reached_capacity = true;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} D2H FIFO reached capacity ({} pages); the host is reading profiler "
            "data slower than the device produces it, so the device holds records back (delayed, not dropped) and "
            "dispatch can stall if this persists",
            dev_state.chip_id,
            available);
    }
}

void RealtimeProfilerReceiver::discard_pending_pages() {
    for (auto& dev_state : devices_) {
        dev_state.socket->discard_pending_pages();
        dev_state.unacked_pages = 0;
    }
}

uint32_t RealtimeProfilerReceiver::drain_device_pages(RealtimeProfilerDevice& dev_state) {
    const uint32_t available = dev_state.socket->pages_available();
    note_fifo_depth(dev_state, available);
    if (available == 0) {
        if (dev_state.unacked_pages != 0) {
            acknowledge(dev_state);
        }
        return 0;
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    dev_state.socket->read(page_buf_.data(), num_pages_to_read, /*notify_sender=*/false);
    dev_state.unacked_pages += num_pages_to_read;
    if (dev_state.unacked_pages >= kAckBatchPages) {
        acknowledge(dev_state);
    }
    publish_pages(dev_state, num_pages_to_read);
    return available;
}

void RealtimeProfilerReceiver::run_receiver_loop() {
    // Sleep between passes over the device FIFOs.
    constexpr std::chrono::microseconds kReceiverPeriod{250};
    // With no consumer registered there is nobody to hand records to, so the FIFOs are acked unread; this period
    // only has to keep them from filling.
    constexpr std::chrono::milliseconds kNoConsumerPeriod{1};
    // Read again without sleeping once any device FIFO holds this many pages.
    constexpr uint32_t kReceiverBacklogPages = kMaxSocketPagesPerRead / 2;
    // Consumers are woken at most this often; the ring holds the records published in between.
    constexpr std::chrono::milliseconds kConsumerWakePeriod{1};
    constexpr auto kFifoPlotInterval = std::chrono::milliseconds(10);
    auto last_fifo_plot = std::chrono::steady_clock::now();
    auto last_consumer_wake = last_fifo_plot;
    const auto loop_start = last_fifo_plot;
    auto last_anchor = last_fifo_plot;
    bool consumers_owed_wake = false;
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        if (consumers_owed_wake && now - last_consumer_wake >= kConsumerWakePeriod) {
            service_->wake_consumers();
            consumers_owed_wake = false;
            last_consumer_wake = now;
        }
        if (receiver_paused_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(kReceiverPeriod);
            continue;
        }
        if (now - last_anchor >= anchor_period(now - loop_start)) {
            last_anchor = now;
            for (auto& dev_state : devices_) {
                anchor_clock_mapping(dev_state);
                service_->tracy().calibrate_device(
                    dev_state.chip_id, dev_state.anchor_host, dev_state.anchor_device, dev_state.frequency);
                publish_device_profiler_sync_anchor(dev_state);
            }
            // The Tracy consumer places zones once an anchor brackets them, so a new anchor can unblock some.
            service_->wake_consumers();
        }
        if (!service_->has_consumers()) {
            discard_pending_pages();
            std::this_thread::sleep_for(kNoConsumerPeriod);
            continue;
        }
        const uint32_t most_pages = drain_all_devices();
        consumers_owed_wake |= most_pages > 0;
        if (now - last_fifo_plot >= kFifoPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(windowed_peak_fifo_pages_));
            windowed_peak_fifo_pages_ = 0;
            last_fifo_plot = now;
        }
        if (most_pages >= kReceiverBacklogPages) {
            continue;
        }
        std::this_thread::sleep_for(kReceiverPeriod);
    }
}

uint32_t RealtimeProfilerReceiver::drain_all_devices() {
    uint32_t most_pages = 0;
    for (auto& dev_state : devices_) {
        most_pages = std::max(most_pages, drain_device_pages(dev_state));
    }
    return most_pages;
}

void RealtimeProfilerReceiver::run_receiver() {
    tracy::SetThreadName("RealtimeProfiler");
#if defined(__linux__)
    ::prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

    run_receiver_loop();
    // Teardown stops the loop only once every push kernel has handed over its last page, so what is left is already
    // in the FIFOs.
    while (drain_all_devices() != 0) {
    }
    log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread stopped");
}

namespace {

// Upper bound on the wait for the push kernel to finish draining.
constexpr auto kShutdownKernelExitGrace = std::chrono::milliseconds(100);
constexpr auto kShutdownKernelExitPollBackoff = std::chrono::microseconds(50);

// Re-writes ring_buffer->terminate as a safety net, so the push kernel delivers the last PCIe page and exits.
void request_kernel_exit(RealtimeProfilerDevice& dev_state) {
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

// Waits, up to `deadline`, for the push kernel to hand over its last PCIe page. The kernel bumps read_index only
// after a page's write has landed, so read_index == write_index means every entry is in the host FIFO.
void wait_for_push_kernel(RealtimeProfilerDevice& dev_state, std::chrono::steady_clock::time_point deadline) {
    const uint32_t indices_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, write_index);
    static_assert(
        offsetof(RtProfilerRingBuffer, read_index) == offsetof(RtProfilerRingBuffer, write_index) + sizeof(uint32_t),
        "write_index and read_index must be adjacent to be read in one shot");
    std::vector<uint32_t> indices(2, 0);
    try {
        while (true) {
            tt::tt_metal::detail::ReadFromDeviceL1(
                dev_state.device,
                dev_state.realtime_profiler_core,
                indices_addr,
                2 * sizeof(uint32_t),
                indices,
                CoreType::WORKER);
            if (indices[0] == indices[1]) {
                log_debug(tt::LogMetal, "[Real-time profiler] Device {} push kernel drained", dev_state.chip_id);
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} push kernel did not drain within {} ms "
                    "(write_index={}, read_index={}); trailing records may be lost",
                    dev_state.chip_id,
                    std::chrono::duration_cast<std::chrono::milliseconds>(kShutdownKernelExitGrace).count(),
                    indices[0],
                    indices[1]);
                return;
            }
            std::this_thread::sleep_for(kShutdownKernelExitPollBackoff);
        }
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Failed to read ring indices while draining device {}: {}",
            dev_state.chip_id,
            e.what());
    }
}

void log_ring_full_waits(const RealtimeProfilerDevice& dev_state) {
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
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Failed to read ring_full_wait_count for device {}: {}",
            dev_state.chip_id,
            e.what());
        return;
    }
    if (full_wait[0] != 0) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} L1 ring hit capacity {} time(s): the path to the host backed up "
            "and profiler records were delayed (not dropped)",
            dev_state.chip_id,
            full_wait[0]);
    }
}

void log_dispatch_stalls(const RealtimeProfilerDevice& dev_state) {
    if (dev_state.dispatch_stall_events != 0) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {}: dispatch stalled {} time(s), {:.2f} us in total, waiting for the "
            "real-time profiler (record ring full); profiler records were delayed, not lost",
            dev_state.chip_id,
            dev_state.dispatch_stall_events,
            cycles_to_us(dev_state.dispatch_stall_cycles, dev_state.frequency));
    }
}

}  // namespace

RealtimeProfilerReceiver::~RealtimeProfilerReceiver() {
    for (auto& dev_state : devices_) {
        request_kernel_exit(dev_state);
    }
    const auto deadline = std::chrono::steady_clock::now() + kShutdownKernelExitGrace;
    for (auto& dev_state : devices_) {
        wait_for_push_kernel(dev_state, deadline);
    }

    stop_.store(true, std::memory_order_release);
    receiver_thread_.join();

    // The receiver has drained every page, so each dispatch-stall marker has been counted.
    for (const auto& dev_state : devices_) {
        log_ring_full_waits(dev_state);
        log_dispatch_stalls(dev_state);
    }
    if (const uint64_t dropped = num_end_before_start_records(); dropped != 0) {
        log_warning(tt::LogMetal, "[Real-time profiler] Dropped {} record(s) that ended before they started", dropped);
    }

    // The receiver thread has stopped publishing; consumers deliver what is left in the ring, then release it.
    service_->detach_producer(ring_);

    for (const auto& dev_state : devices_) {
        service_->tracy().remove_device(dev_state.chip_id);
    }
    // The Tracy consumer holds back the last zones until it is told the device is gone; let it place them.
    service_->wake_consumers();
    service_->tracy().wait_for_events(std::chrono::seconds(2));
    data_collector_->RemoveProgramMetadataReader();
}

}  // namespace tt::tt_metal
