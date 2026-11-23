// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <enchantum/enchantum.hpp>
#include <fmt/core.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/math.hpp>
#include <internal/mesh_device.hpp>
#include <tt_metal.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/xy_pair.hpp>

#include <common/TracySystem.hpp>
#include <llrt/tt_cluster.hpp>
#include <tracy/TracyTTDevice.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/hal.hpp"
#include "program/program_impl.hpp"
#include "program/slow_dispatch.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_tracy_consumer.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"                // tt::tt_metal::SyncInfo, DeviceProfiler
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"  // ProfilerStateManager
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal::distributed {

namespace {

// Minimum wall time between full init calibrations (run_sync + constructor SYNC_CHECK) and
// between finish-path sync checks, per physical chip. Matches the finish-path throttle.
constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

constexpr auto kFinishSyncRequestDelay = std::chrono::milliseconds(5);
constexpr auto kFinishSyncResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kSyncResponsePollBackoff = std::chrono::microseconds(100);

// Last full init sync per chip, process-wide, to avoid repeating ~0.5s run_sync on every mesh open.
std::mutex g_rt_profiler_init_sync_mu;
std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> g_rt_profiler_last_init_sync_by_chip;

// Sync marker ID — must match device-side REALTIME_PROFILER_SYNC_MARKER_ID.
constexpr uint32_t REALTIME_PROFILER_SYNC_MARKER_ID = 0xFFFFFFFF;

// Dispatch-stall marker ID — must match device-side REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID. Page words 0-1:
// device time the stall ended, word 2: its length in device cycles.
constexpr uint32_t REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID = 0xFFFFFFFE;

// Device cycles to microseconds; frequency is in cycles per nanosecond (0 before the first sync).
double cycles_to_us(uint64_t cycles, double frequency) {
    return frequency > 0.0 ? static_cast<double>(cycles) / (frequency * 1000.0) : 0.0;
}

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// Host clock for the sync handshake; falls back to steady_clock since Tracy stubs TracyGetCpuTime to 0 when disabled
// (which would stall the device).
inline int64_t rt_profiler_host_ticks() {
#ifdef TRACY_ENABLE
    return TracyGetCpuTime();
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

inline double rt_profiler_ns_per_tick() {
#ifdef TRACY_ENABLE
    return TracyGetTimerMul();
#else
    return 1.0;
#endif
}

// Concurrent host-device sync per device (distinct PCIe paths / sockets). Uses up to
// hardware_concurrency workers; single-threaded when only one task or concurrency unknown.
template <typename Fn>
void parallel_for_each_device_index(const std::vector<size_t>& indices, Fn&& fn) {
    if (indices.empty()) {
        return;
    }
    // Single std::forward: cppcoreguidelines-missing-std-forward; callable is then invoked
    // many times (not forwarding the parameter each time — bugprone-use-after-move).
    std::decay_t<Fn> callable = std::forward<Fn>(fn);
    auto invoke = [&callable](size_t di) {
        try {
            callable(di);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal, "[Real-time profiler] Per-device init sync failed, skipping device: {}", e.what());
        } catch (...) {
            log_warning(
                tt::LogMetal, "[Real-time profiler] Per-device init sync failed, skipping device (unknown error)");
        }
    };
    const unsigned hc = std::thread::hardware_concurrency();
    const size_t worker_count = std::min(indices.size(), static_cast<size_t>(std::max(1u, hc)));
    if (worker_count <= 1) {
        for (size_t di : indices) {
            invoke(di);
        }
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t w = 0; w < worker_count; ++w) {
        workers.emplace_back([&]() {
            while (true) {
                const size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= indices.size()) {
                    break;
                }
                invoke(indices[k]);
            }
        });
    }
    for (auto& t : workers) {
        t.join();
    }
}

}  // namespace

uint32_t RealtimeProfilerManager::host_fifo_capacity_pages() const { return RealtimeProfilerRuntimeSizes::fifo_pages; }

uint32_t RealtimeProfilerManager::ring_full_wait_count() const {
    uint32_t peak = 0;
    for (const auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer == 0 || !dev_state.device) {
            continue;
        }
        const uint32_t addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, ring_full_wait_count);
        std::vector<uint32_t> value(1, 0);
        tt::tt_metal::detail::ReadFromDeviceL1(
            dev_state.device, dev_state.realtime_profiler_core, addr, sizeof(uint32_t), value, CoreType::WORKER);
        peak = std::max(peak, value[0]);
    }
    return peak;
}

uint32_t RealtimeProfilerManager::record_ring_full_wait_count() const {
    uint32_t peak = 0;
    for (const auto& dev_state : devices_) {
        if (!dev_state.dispatch_s_core.has_value() || !dev_state.device) {
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

void RealtimeProfilerManager::report_dispatch_stall(
    DeviceState& dev_state, uint64_t stall_end_timestamp, uint32_t stall_cycles) {
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
            cycles_to_us(stall_cycles, dev_state.sync_frequency));
    }
    tracy_handler_->PushDispatchStallMarker(
        dev_state.chip_id, stall_end_timestamp, stall_cycles, dev_state.sync_frequency);
}

void RealtimeProfilerManager::publish_pages(
    DeviceState& dev_state,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<tt::ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    auto is_record = [](const uint32_t* page) {
        return page[2] != 0 && page[3] != REALTIME_PROFILER_SYNC_MARKER_ID &&
               page[3] != REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID;
    };
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double sync_frequency = dev_state.sync_frequency;
    const DataCollector* const data_collector = data_collector_;
    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = page_buf + page * kPageWords;
        if (rp[3] == REALTIME_PROFILER_DISPATCH_STALL_MARKER_ID) {
            report_dispatch_stall(dev_state, (static_cast<uint64_t>(rp[0]) << 32) | rp[1], rp[2]);
            continue;
        }
        if (!is_record(rp)) {
            continue;
        }
        records.emplace_back(
            rp[2],
            chip_id,
            (static_cast<uint64_t>(rp[0]) << 32) | rp[1],
            (static_cast<uint64_t>(rp[4]) << 32) | rp[5],
            sync_frequency,
            data_collector->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])));
    }
    if (records.empty()) {
        return;
    }
    num_published_records_.fetch_add(records.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_->writer().publish_batch(std::span<const tt::ProgramRealtimeRecord>(records));
}

bool RealtimeProfilerManager::has_active_finish_sync() const {
    for (const auto& dev_state : devices_) {
        if (dev_state.finish_sync_phase != DeviceState::FinishSyncPhase::Idle) {
            return true;
        }
    }
    return false;
}

void RealtimeProfilerManager::write_sync_request(RealtimeProfilerManager::DeviceState& dev_state, SyncRequest value) {
    std::vector<uint32_t> data = {value};
    tt::tt_metal::detail::WriteToDeviceL1(
        dev_state.device, dev_state.realtime_profiler_core, dev_state.sync_request_addr, data, CoreType::WORKER);
}

void RealtimeProfilerManager::start_finish_syncs(std::chrono::steady_clock::time_point now) {
    if (!finish_sync_requested_.load(std::memory_order_acquire)) {
        return;
    }

    bool started = false;
    for (auto& dev_state : devices_) {
        if (dev_state.finish_sync_phase != DeviceState::FinishSyncPhase::Idle) {
            continue;
        }
        const bool interval_elapsed = !dev_state.last_finish_sync_at.has_value() ||
                                      now - *dev_state.last_finish_sync_at >= kRtProfilerMinSyncInterval;
        if (!interval_elapsed && !dev_state.pending_first_unthrottled_finish_sync) {
            continue;
        }
        try {
            write_sync_request(dev_state, SyncRequest::Set);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Failed to start sync for device {}: {}",
                dev_state.chip_id,
                e.what());
            continue;
        }
        dev_state.finish_sync_request_at = now;
        dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::AwaitingDelay;
        started = true;
    }
    finish_sync_busy_.store(started || has_active_finish_sync(), std::memory_order_release);
    finish_sync_requested_.store(false, std::memory_order_release);
    notify_finish_sync_waiters();
}

void RealtimeProfilerManager::advance_finish_sync(DeviceState& dev_state, std::chrono::steady_clock::time_point now) {
    switch (dev_state.finish_sync_phase) {
        case DeviceState::FinishSyncPhase::Idle: return;
        case DeviceState::FinishSyncPhase::AwaitingDelay: {
            if (now - dev_state.finish_sync_request_at < kFinishSyncRequestDelay) {
                return;
            }
            dev_state.sync_host_time_before = rt_profiler_host_ticks();
            std::vector<uint32_t> host_time_data = {
                static_cast<uint32_t>(dev_state.sync_host_time_before & 0xFFFFFFFF)};
            TracyMessageL("FINISH_SYNC");
            tt::tt_metal::detail::WriteToDeviceL1(
                dev_state.device,
                dev_state.realtime_profiler_core,
                dev_state.sync_host_ts_addr,
                host_time_data,
                CoreType::WORKER);
            dev_state.finish_sync_deadline = now + kFinishSyncResponseTimeout;
            dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::AwaitingResponse;
            return;
        }
        case DeviceState::FinishSyncPhase::AwaitingResponse:
            if (now > dev_state.finish_sync_deadline) {
                log_warning(tt::LogMetal, "[Real-time profiler] Sync check timed out for device {}", dev_state.chip_id);
                write_sync_request(dev_state, SyncRequest::Clear);
                dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::Idle;
                finish_sync_busy_.store(has_active_finish_sync(), std::memory_order_release);
                notify_finish_sync_waiters();
            }
            return;
    }
}

void RealtimeProfilerManager::service_finish_sync(std::chrono::steady_clock::time_point now, bool allow_start) {
    if (allow_start) {
        start_finish_syncs(now);
    }
    if (!finish_sync_busy_.load(std::memory_order_acquire)) {
        return;
    }
    for (auto& dev_state : devices_) {
        try {
            advance_finish_sync(dev_state, now);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Exception advancing sync for device {}: {}",
                dev_state.chip_id,
                e.what());
        }
    }
}

RealtimeProfilerManager::RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device) :
    context_id_(mesh_device->impl().get_context_id()) {
    initialize_devices(mesh_device);

    if (devices_.empty()) {
        log_debug(
            tt::LogMetal, "[Real-time profiler] No local devices found in mesh, skipping real-time profiler setup");
        return;
    }

    const size_t max_consumer_batch_records =
        std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * devices_.size());
    ring_.emplace(std::min(kMaxRingCapacity, max_consumer_batch_records * kRingHeadroomBatches));

    for (const auto& dev_state : devices_) {
        tt::NotifyProgramRealtimeProfilerActivated(context_id_, dev_state.chip_id);
    }

    run_init_sync();

    for (auto& dev_state : devices_) {
        dev_state.pending_first_unthrottled_finish_sync = true;
    }

    DataCollector* data_collector = MetalContext::instance(context_id_).data_collector().get();
    data_collector->AttachRealtimeProfilerCallbackListener(this);
    data_collector_ = data_collector;

    // Background receiver thread that polls all device sockets round-robin
    receiver_thread_ = std::thread(&RealtimeProfilerManager::run_receiver, this);
}

void RealtimeProfilerManager::run_init_sync() {
    constexpr uint32_t kInitSyncMaxRetries = 3;
    constexpr auto kInitSyncRetryDelay = std::chrono::milliseconds(500);
    constexpr auto kConstructorSyncCheckDelay = std::chrono::milliseconds(10);
    constexpr auto kConstructorSyncCheckTimeout = std::chrono::milliseconds(3000);
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto init_throttle_now = std::chrono::steady_clock::now();
    std::vector<bool> skip_init_sync_check(devices_.size(), false);
    std::vector<size_t> init_run_sync_indices;
    init_run_sync_indices.reserve(devices_.size());

    // Run our own host-device sync (device profiler's SyncInfo masks the high word to 12 bits, shifting RT zones by
    // hours); skip recently init-synced chips.
    for (size_t di = 0; di < devices_.size(); ++di) {
        auto& dev_state = devices_[di];
        bool throttle_skip = false;
        {
            std::lock_guard<std::mutex> lock(g_rt_profiler_init_sync_mu);
            const auto it = g_rt_profiler_last_init_sync_by_chip.find(dev_state.chip_id);
            if (it != g_rt_profiler_last_init_sync_by_chip.end() &&
                init_throttle_now - it->second < kRtProfilerMinSyncInterval) {
                throttle_skip = true;
            }
        }

        if (throttle_skip) {
            const int64_t host_start = rt_profiler_host_ticks();
            dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
            dev_state.first_timestamp = 0;
            dev_state.sync_host_start = host_start;
            dev_state.last_finish_sync_at = init_throttle_now;
            skip_init_sync_check[di] = true;
            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: skipping init run_sync and constructor SYNC_CHECK "
                "(last init sync within {}s; using AICLK frequency fallback)",
                dev_state.chip_id,
                static_cast<int>(kRtProfilerMinSyncInterval.count()));
            continue;
        }

        init_run_sync_indices.push_back(di);
    }

    parallel_for_each_device_index(init_run_sync_indices, [&](size_t di) {
        auto& dev_state = devices_[di];
        for (uint32_t attempt = 0; attempt <= kInitSyncMaxRetries; attempt++) {
            if (attempt > 0) {
                log_debug(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} sync retry {}/{}",
                    dev_state.chip_id,
                    attempt,
                    kInitSyncMaxRetries);
                std::this_thread::sleep_for(kInitSyncRetryDelay);
            }
            run_sync(dev_state, 100);
            if (dev_state.first_timestamp != 0) {
                break;
            }
        }
        if (dev_state.first_timestamp != 0) {
            std::lock_guard<std::mutex> lock(g_rt_profiler_init_sync_mu);
            g_rt_profiler_last_init_sync_by_chip[dev_state.chip_id] = std::chrono::steady_clock::now();
        }
    });

    tracy_handler_ = std::make_unique<RealtimeProfilerTracyHandler>();
    for (const auto& dev_state : devices_) {
        tracy_handler_->AddDevice(
            dev_state.chip_id,
            dev_state.sync_host_start,
            static_cast<double>(dev_state.first_timestamp),
            dev_state.sync_frequency);
        publish_device_profiler_sync_anchor(
            dev_state.chip_id,
            static_cast<double>(dev_state.sync_host_start),
            static_cast<double>(dev_state.first_timestamp),
            dev_state.sync_frequency,
            dev_state.realtime_profiler_core.str());
    }

    // Emit paired host+device SYNC_CHECK markers; their horizontal distance in Tracy is the sync error.
    std::vector<size_t> init_sync_check_indices;
    init_sync_check_indices.reserve(devices_.size());
    for (size_t di = 0; di < devices_.size(); ++di) {
        if (!skip_init_sync_check[di]) {
            init_sync_check_indices.push_back(di);
        }
    }
    parallel_for_each_device_index(init_sync_check_indices, [&](size_t di) {
        auto& dev_state = devices_[di];
        write_sync_request(dev_state, SyncRequest::Set);

        std::this_thread::sleep_for(kConstructorSyncCheckDelay);

        // Capture host TSC, emit Tracy message, then PCIe write; CalibrateDevice must precede PushSyncCheckMarker or
        // skew exceeds the ±10µs test bound.
        int64_t sync_check_host_anchor = rt_profiler_host_ticks();
        uint32_t host_time_id = 0x5C5C5C5C;
        std::vector<uint32_t> host_time_data = {host_time_id};
        TracyMessageL("SYNC_CHECK");
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_host_ts_addr,
            host_time_data,
            CoreType::WORKER);

        auto sc_deadline = std::chrono::steady_clock::now() + kConstructorSyncCheckTimeout;
        bool sc_got_response = false;
        while (std::chrono::steady_clock::now() < sc_deadline) {
            if (dev_state.socket->pages_available() > 0) {
                sc_got_response = true;
                break;
            }
            std::this_thread::sleep_for(kSyncResponsePollBackoff);
        }

        write_sync_request(dev_state, SyncRequest::Clear);

        if (sc_got_response) {
            std::vector<uint32_t> sync_page(RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t));
            dev_state.socket->read(sync_page.data(), 1);
            uint64_t device_time = (static_cast<uint64_t>(sync_page[0]) << 32) | sync_page[1];

            tracy_handler_->CalibrateDevice(
                dev_state.chip_id, sync_check_host_anchor, device_time, dev_state.sync_frequency);
            tracy_handler_->PushSyncCheckMarker(dev_state.chip_id, device_time, dev_state.sync_frequency);
            publish_device_profiler_sync_anchor(
                dev_state.chip_id,
                static_cast<double>(sync_check_host_anchor),
                static_cast<double>(device_time),
                dev_state.sync_frequency,
                dev_state.realtime_profiler_core.str());

            dev_state.last_finish_sync_at = std::chrono::steady_clock::now();

            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {} sync check: device_time={} cycles",
                dev_state.chip_id,
                device_time);
        } else {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} sync check timed out after {}ms, skipping",
                dev_state.chip_id,
                kConstructorSyncCheckTimeout.count());
        }
    });
}

uint32_t RealtimeProfilerManager::drain_device_pages(
    DeviceState& dev_state,
    bool scan_sync_marker,
    std::vector<uint32_t>& page_buf,
    std::vector<tt::ProgramRealtimeRecord>& record_buf) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    uint32_t available = dev_state.socket->pages_available();
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
    if (available == 0) {
        return 0;
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    dev_state.socket->read(page_buf.data(), num_pages_to_read);

    if (scan_sync_marker && dev_state.finish_sync_phase == DeviceState::FinishSyncPhase::AwaitingResponse) {
        for (uint32_t page = 0; page < num_pages_to_read; ++page) {
            const uint32_t* read_ptr = page_buf.data() + page * kPageWords;
            if (read_ptr[3] != REALTIME_PROFILER_SYNC_MARKER_ID) {
                continue;
            }
            const uint64_t device_time = (static_cast<uint64_t>(read_ptr[0]) << 32) | read_ptr[1];
            tracy_handler_->CalibrateDevice(
                dev_state.chip_id, dev_state.sync_host_time_before, device_time, dev_state.sync_frequency);
            tracy_handler_->PushSyncCheckMarker(dev_state.chip_id, device_time, dev_state.sync_frequency);
            publish_device_profiler_sync_anchor(
                dev_state.chip_id,
                static_cast<double>(dev_state.sync_host_time_before),
                static_cast<double>(device_time),
                dev_state.sync_frequency,
                dev_state.realtime_profiler_core.str());
            dev_state.last_finish_sync_at = std::chrono::steady_clock::now();
            dev_state.pending_first_unthrottled_finish_sync = false;
            write_sync_request(dev_state, SyncRequest::Clear);
            dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::Idle;
            finish_sync_busy_.store(has_active_finish_sync(), std::memory_order_release);
            notify_finish_sync_waiters();
            break;
        }
    }
    publish_pages(dev_state, page_buf.data(), num_pages_to_read, record_buf);
    return num_pages_to_read;
}

uint64_t RealtimeProfilerManager::run_receiver_loop() {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * kPageWords);
    std::vector<tt::ProgramRealtimeRecord> record_buf;
    record_buf.reserve(kMaxSocketPagesPerRead);
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    constexpr auto kFifoPlotInterval = std::chrono::milliseconds(10);
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;
    auto last_fifo_plot = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_acquire)) {
        if (receiver_paused_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(kReceiverMaxBackoff);
            continue;
        }
        const bool scan_sync_marker = finish_sync_busy_.load(std::memory_order_acquire);
        const uint32_t num_pages = drain_all_devices(scan_sync_marker, page_buf, record_buf);
        num_pages_received += num_pages;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_fifo_plot >= kFifoPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(windowed_peak_fifo_pages_));
            windowed_peak_fifo_pages_ = 0;
            last_fifo_plot = now;
        }
        const bool sync_requested = finish_sync_requested_.load(std::memory_order_acquire);
        if (scan_sync_marker || sync_requested) {
            service_finish_sync(now, sync_requested);
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

uint32_t RealtimeProfilerManager::drain_all_devices(
    bool scan_sync_marker, std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf) {
    uint32_t num_pages = 0;
    for (auto& dev_state : devices_) {
        try {
            num_pages += drain_device_pages(dev_state, scan_sync_marker, page_buf, record_buf);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal, "[Real-time profiler] Exception draining device {}: {}", dev_state.chip_id, e.what());
        }
    }
    if (num_pages > 0) {
        ring_->writer().wake_readers();
    }
    return num_pages;
}

uint64_t RealtimeProfilerManager::drain_receiver_on_shutdown() {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * kPageWords);
    std::vector<tt::ProgramRealtimeRecord> record_buf;
    record_buf.reserve(kMaxSocketPagesPerRead);
    constexpr uint32_t kShutdownDrainQuietRounds = 10;
    constexpr auto kShutdownDrainQuietBackoff = std::chrono::milliseconds(1);
    uint64_t num_pages_drained = 0;
    uint32_t quiet_rounds = 0;
    while (quiet_rounds < kShutdownDrainQuietRounds) {
        const bool scan_sync_marker = finish_sync_busy_.load(std::memory_order_acquire);
        const uint32_t num_pages = drain_all_devices(scan_sync_marker, page_buf, record_buf);
        if (num_pages != 0) {
            num_pages_drained += num_pages;
            quiet_rounds = 0;
        } else {
            quiet_rounds++;
            std::this_thread::sleep_for(kShutdownDrainQuietBackoff);
        }
    }
    return num_pages_drained;
}

void RealtimeProfilerManager::run_receiver() {
    tracy::SetThreadName("RealtimeProfiler");
#if defined(__linux__)
    ::prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

    const uint64_t num_pages_received = run_receiver_loop();
    const uint64_t num_pages_drained = drain_receiver_on_shutdown();

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown)",
        num_pages_received + num_pages_drained,
        num_pages_drained);
}

RealtimeProfilerManager::~RealtimeProfilerManager() { shutdown(); }

void RealtimeProfilerManager::shutdown() {
    // Upper bound on the wait for the push kernel to finish draining; see the poll below.
    constexpr auto kShutdownKernelExitGrace = std::chrono::milliseconds(100);
    constexpr auto kShutdownKernelExitPollBackoff = std::chrono::microseconds(50);
    MetalContext::instance(context_id_).data_collector()->DetachRealtimeProfilerCallbackListener(this);
    // The push kernel can only finish once the receiver reads again.
    receiver_paused_.store(false, std::memory_order_release);

    // Re-write ring_buffer->terminate as a safety net, then let the push kernel deliver the last PCIe page.
    for (auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer != 0 && dev_state.device) {
            const uint32_t terminate_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, terminate);
            std::vector<uint32_t> terminate_flag = {1};
            try {
                write_sync_request(dev_state, SyncRequest::Clear);
                tt::tt_metal::detail::WriteToDeviceL1(
                    dev_state.device,
                    dev_state.realtime_profiler_core,
                    terminate_addr,
                    terminate_flag,
                    CoreType::WORKER);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to write terminate flag for device {}: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
    }
    // Wait for the push kernel to hand over its last PCIe page and exit. It drains the ring and returns once
    // it observes terminate with an empty ring, and it bumps read_index only after that page's write barrier
    // has retired, so read_index == write_index means every entry has landed in the host ring. Poll for that
    // equality rather than waiting out kShutdownKernelExitGrace: draining takes microseconds, and shutdown()
    // runs on every close_device, where suites with a function-scoped `device` fixture pay it per test.
    const auto deadline = std::chrono::steady_clock::now() + kShutdownKernelExitGrace;
    for (auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer == 0 || !dev_state.device) {
            continue;
        }
        const uint32_t indices_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, write_index);
        static_assert(
            offsetof(RtProfilerRingBuffer, read_index) ==
                offsetof(RtProfilerRingBuffer, write_index) + sizeof(uint32_t),
            "write_index and read_index must be adjacent to be read in one shot");
        bool drained = false;
        while (true) {
            std::vector<uint32_t> indices(2, 0);
            try {
                tt::tt_metal::detail::ReadFromDeviceL1(
                    dev_state.device,
                    dev_state.realtime_profiler_core,
                    indices_addr,
                    2 * sizeof(uint32_t),
                    indices,
                    CoreType::WORKER);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to read ring indices while draining device {}: {}",
                    dev_state.chip_id,
                    e.what());
                break;
            }
            if (indices[0] == indices[1]) {
                drained = true;
                break;
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
                break;
            }
            std::this_thread::sleep_for(kShutdownKernelExitPollBackoff);
        }
        if (drained) {
            log_debug(tt::LogMetal, "[Real-time profiler] Device {} push kernel drained", dev_state.chip_id);
        }
    }

    if (receiver_thread_.joinable()) {
        stop_.store(true, std::memory_order_release);
        notify_finish_sync_waiters();
        receiver_thread_.join();
    }

    for (const auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer == 0 || !dev_state.device) {
            continue;
        }
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
                    "[Real-time profiler] Device {} L1 ring hit capacity {} time(s): the path to the host backed up "
                    "and profiler records were delayed (not dropped)",
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

    // The receiver has drained every page, so each dispatch-stall marker has been counted.
    for (const auto& dev_state : devices_) {
        if (dev_state.dispatch_stall_events != 0) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: dispatch stalled {} time(s), {:.2f} us in total, waiting for the "
                "real-time profiler (record ring full); profiler records were delayed, not lost",
                dev_state.chip_id,
                dev_state.dispatch_stall_events,
                cycles_to_us(dev_state.dispatch_stall_cycles, dev_state.sync_frequency));
        }
    }

    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        for (auto& [handle, consumer] : consumers_) {
            stop_consumer(*consumer, ConsumerStopMode::DrainThenStop);
        }
        consumers_.clear();
    }

    tracy_handler_.reset();
    // Clear activation state before destroying per-device records so concurrent
    // tt::IsProgramRealtimeProfilerActive() queries don't observe a chip mid-shutdown.
    for (const auto& dev_state : devices_) {
        tt::NotifyProgramRealtimeProfilerDeactivated(context_id_, dev_state.chip_id);
    }
    devices_.clear();
}

void RealtimeProfilerManager::run_sync(DeviceState& dev_state, uint32_t num_samples) {
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr auto kRunSyncReadTimeout = std::chrono::milliseconds(2000);
    constexpr uint32_t kRunSyncMaxConsecutiveTimeouts = 3;
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    int64_t host_start_time = rt_profiler_host_ticks();

    struct SyncSample {
        int64_t host_time;     // Full 64-bit host TSC ticks relative to host_start_time
        uint64_t device_time;  // Device wall clock cycles
    };
    std::vector<SyncSample> samples;
    samples.reserve(num_samples);

    // Discard pre-existing pages before sync (their PCIe-mapped bytes can be undefined on a fresh MeshDevice);
    // discard_pending_pages rebases bytes_acked and notifies the device.
    constexpr uint32_t kSyncPageWords = 64 / sizeof(uint32_t);
    uint32_t stale_pages = dev_state.socket->discard_pending_pages();
    if (stale_pages > 0) {
        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {} discarded {} stale pages before sync",
            dev_state.chip_id,
            stale_pages);
    }

    write_sync_request(dev_state, SyncRequest::Set);

    std::this_thread::sleep_for(kRunSyncSettleDelay);

    uint32_t consecutive_timeouts = 0;

    for (uint32_t i = 0; i < num_samples + 1; i++) {
        std::this_thread::sleep_for(kRunSyncSampleInterval);

        // Send truncated 32-bit value as echo identifier for pairing.
        int64_t host_before = rt_profiler_host_ticks() - host_start_time;
        uint32_t host_time_id = static_cast<uint32_t>(host_before);
        std::vector<uint32_t> host_time_data = {host_time_id};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_host_ts_addr,
            host_time_data,
            CoreType::WORKER);

        auto deadline = std::chrono::steady_clock::now() + kRunSyncReadTimeout;
        bool got_response = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (dev_state.socket->pages_available() > 0) {
                got_response = true;
                break;
            }
            std::this_thread::sleep_for(kSyncResponsePollBackoff);
        }

        if (!got_response) {
            consecutive_timeouts++;
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} sync sample {}/{} timed out after {}ms "
                "(consecutive timeouts: {}/{})",
                dev_state.chip_id,
                i,
                num_samples,
                kRunSyncReadTimeout.count(),
                consecutive_timeouts,
                kRunSyncMaxConsecutiveTimeouts);
            if (consecutive_timeouts >= kRunSyncMaxConsecutiveTimeouts) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} sync aborted: {} consecutive timeouts. "
                    "Device kernel may not be responding (check DPRINT output).",
                    dev_state.chip_id,
                    consecutive_timeouts);
                break;
            }
            continue;
        }

        consecutive_timeouts = 0;
        std::vector<uint32_t> sync_data(kSyncPageWords);
        dev_state.socket->read(sync_data.data(), 1);
        uint64_t device_time = (static_cast<uint64_t>(sync_data[0]) << 32) | sync_data[1];
        uint32_t echoed_host_time = sync_data[2];
        uint32_t marker = sync_data[3];

        // Discard first sample - can be very off due to cold PCIe path.
        if (i == 0) {
            continue;
        }

        // Use host_before (not midpoint) because H2D and D2H latencies are asymmetric;
        // host_before brackets the device-side capture within ~2µs.
        if (marker == REALTIME_PROFILER_SYNC_MARKER_ID && echoed_host_time == host_time_id) {
            samples.emplace_back(host_before, device_time);
        }
    }

    write_sync_request(dev_state, SyncRequest::Clear);

    // Mean-centered linear regression for slope (device cycles per TSC tick); centering avoids catastrophic
    // cancellation at absolute-timestamp magnitudes.
    if (samples.size() >= 2) {
        const double n = static_cast<double>(samples.size());
        const double tracy_ratio = rt_profiler_ns_per_tick();

        double host_mean = 0.0;
        double device_mean = 0.0;
        for (const auto& s : samples) {
            host_mean += static_cast<double>(s.host_time);
            device_mean += static_cast<double>(s.device_time);
        }
        host_mean /= n;
        device_mean /= n;

        double num = 0.0;
        double den = 0.0;
        for (const auto& s : samples) {
            double dx = static_cast<double>(s.host_time) - host_mean;
            double dy = static_cast<double>(s.device_time) - device_mean;
            num += dx * dy;
            den += dx * dx;
        }

        if (std::abs(den) > 1e-10) {
            // slope = device_cycles per host_TSC_tick
            // frequency = slope / tracy_ratio = device_cycles per nanosecond (GHz)
            double slope = num / den;
            dev_state.sync_frequency = slope / tracy_ratio;
        } else {
            dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
        }

        // Intercept via means: intercept = ȳ - slope * x̄ = device cycle count at host_time = 0.
        double slope = dev_state.sync_frequency * tracy_ratio;
        double intercept = device_mean - slope * host_mean;
        dev_state.first_timestamp = static_cast<uint64_t>(intercept);
        dev_state.sync_host_start = host_start_time;

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "device_time_at_sync={} cycles",
            dev_state.chip_id,
            samples.size(),
            dev_state.sync_frequency,
            dev_state.first_timestamp);
        // Device-profiler sync anchor is published in lockstep with the rt calibration sites, not here -- see
        // publish_device_profiler_sync_anchor().
    } else {
        dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
        dev_state.first_timestamp = 0;
        dev_state.sync_host_start = host_start_time;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using default frequency",
            dev_state.chip_id);
    }
}

void RealtimeProfilerManager::publish_device_profiler_sync_anchor(
    uint32_t chip_id, double host_anchor, double device_anchor, double frequency, const std::string& core_label) {
    // Accumulate-only: there the device profiler skips its own sync and borrows the rt fit; otherwise it runs its own
    // sync so leave realtime_sync_line unset.
    if (!MetalContext::instance(context_id_).rtoptions().get_profiler_accumulate()) {
        return;
    }
    // Pass the raw anchor (host TSC, device cycle, frequency), not a SyncInfo: the device profiler keeps its own worker
    // anchor and only adopts our host<->device mapping. Valid because all cores share one wall clock.
    auto& psm = MetalContext::instance(context_id_).profiler_state_manager();
    if (!psm || !psm->device_profiler_map.contains(chip_id)) {
        return;
    }
    {
        std::lock_guard<std::recursive_mutex> map_lock(psm->device_profiler_map_mutex);
        psm->device_profiler_map.at(chip_id).realtime_sync_line =
            tt::tt_metal::DeviceProfiler::RealtimeSyncLine{host_anchor, device_anchor, frequency};
    }
    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device-profiler clock anchor for device {} core {}: "
        "host_anchor={:.0f}, device_anchor={:.0f}, freq={:.6f} GHz",
        chip_id,
        core_label,
        host_anchor,
        device_anchor,
        frequency);
}

void RealtimeProfilerManager::notify_finish_sync_waiters() {
    std::lock_guard<std::mutex> lock(finish_sync_wait_mu_);
    finish_sync_cv_.notify_all();
}

void RealtimeProfilerManager::trigger_sync_check() {
    constexpr auto kFinishSyncWaitSlack = std::chrono::seconds(1);
    if (devices_.empty() || !tracy_handler_) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point last{
        std::chrono::steady_clock::duration{last_sync_request_at_.load(std::memory_order_relaxed)}};
    if (now - last < kRtProfilerMinSyncInterval) {
        return;
    }
    last_sync_request_at_.store(now.time_since_epoch().count(), std::memory_order_relaxed);

    finish_sync_requested_.store(true, std::memory_order_release);
    const auto deadline = now + kFinishSyncRequestDelay + kFinishSyncResponseTimeout + kFinishSyncWaitSlack;
    {
        std::unique_lock<std::mutex> lock(finish_sync_wait_mu_);
        finish_sync_cv_.wait_until(lock, deadline, [this] {
            return stop_.load(std::memory_order_acquire) || (!finish_sync_requested_.load(std::memory_order_acquire) &&
                                                             !finish_sync_busy_.load(std::memory_order_acquire));
        });
    }
    if (!stop_.load(std::memory_order_acquire) &&
        (finish_sync_requested_.load(std::memory_order_acquire) || finish_sync_busy_.load(std::memory_order_acquire))) {
        log_warning(tt::LogMetal, "[Real-time profiler] Timed out waiting for finish-path sync to complete");
    }
}

}  // namespace tt::tt_metal::distributed
