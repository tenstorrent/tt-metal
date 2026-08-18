// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/realtime_profiler_manager.hpp"

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

#include <fmt/core.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt_metal.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <common/TracySystem.hpp>
#include <common/TracyTTDeviceData.hpp>
#include <llrt/tt_cluster.hpp>
#include <tracy/TracyTTDevice.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/hal.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/hw/inc/hostdev/realtime_profiler_protocol_common.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/dispatch/realtime_profiler_tracy_handler.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"                // tt::tt_metal::SyncInfo, DeviceProfiler
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"  // ProfilerStateManager
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal::distributed {

namespace {

template <typename Function>
class ScopeExit {
public:
    explicit ScopeExit(Function function) : function_(std::move(function)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    ~ScopeExit() {
        if (active_) {
            function_();
        }
    }

    void release() { active_ = false; }

private:
    Function function_;
    bool active_ = true;
};

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

// Real-time profiler runtime constants. On-device L1 layout sizes are reused from
// realtime_profiler_ring_buffer.hpp so host and device share a single source of truth.
struct RealtimeProfilerRuntimeSizes {
    static constexpr uint32_t fifo_pages = RT_PROFILER_HOST_FIFO_PAGES;  // host D2H FIFO depth, in pages
    static constexpr uint32_t page_size = RT_PROFILER_ENTRY_SIZE;  // host page size == ring entry size
    static constexpr uint32_t fifo_size = fifo_pages * page_size;  // pinned-host FIFO, in bytes (2 MiB)
    static constexpr uint32_t core_l1_size = sizeof(RealtimeProfilerCoreL1);
};

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// Compute the RT-profiler L1 carve-out addresses from a base anchored past UNRESERVED (outside the user-space
// allocator).
inline RealtimeProfilerCoreL1Addrs compute_rt_profiler_core_l1_addrs(uint32_t base) {
    return {
        .base = base,
        .ring_buffer = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, ring)),
        .socket_config = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, socket_config)),
    };
}

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

RealtimeProfilerManager::DeviceState::DeviceState() = default;
RealtimeProfilerManager::DeviceState::~DeviceState() = default;
RealtimeProfilerManager::DeviceState::DeviceState(DeviceState&&) noexcept = default;

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

std::vector<tt::ProgramRealtimeProfilerDeviceCapability> RealtimeProfilerManager::get_device_capabilities() const {
    std::vector<tt::ProgramRealtimeProfilerDeviceCapability> capabilities;
    capabilities.reserve(devices_.size());
    for (const auto& dev_state : devices_) {
        tt::ProgramRealtimeProfilerDeviceCapability capability{
            .chip_id = dev_state.chip_id,
            .active = true,
            .inactive_reason = experimental::ProgramRealtimeProfilerInactiveReason::None,
        };
        if (dev_state.device != nullptr && dev_state.dispatch_s_profiler_msg_addr != 0) {
            try {
                constexpr uint32_t kLossWordCount = 9;
                std::vector<uint32_t> words(kLossWordCount, 0);
                tt::tt_metal::detail::ReadFromDeviceL1(
                    dev_state.device,
                    dev_state.dispatch_s_core,
                    dev_state.dispatch_s_profiler_msg_addr + offsetof(realtime_profiler_msg_t, loss_descriptor_full),
                    kLossWordCount * sizeof(uint32_t),
                    words,
                    CoreType::WORKER);
                capability.loss.descriptor_full = words[0];
                capability.loss.unsupported_launch = words[1];
                capability.loss.terminal_descriptor = words[2];
                capability.loss.reset_descriptor = words[3];
                capability.loss.observer_coalesced = words[4];
                capability.loss.stuck_head = words[5];
                capability.loss.completed_record = words[6];
                capability.loss.terminal_record = words[7];
                capability.loss.observer_stop_timeout = words[8];
                if (dev_state.dispatch_s_profiler_msg_addr != 0) {
                    std::vector<uint32_t> device_ring_loss(1, 0);
                    tt::tt_metal::detail::ReadFromDeviceL1(
                        dev_state.device,
                        dev_state.realtime_profiler_core,
                        dev_state.dispatch_s_profiler_msg_addr + offsetof(realtime_profiler_msg_t, loss_device_ring),
                        sizeof(uint32_t),
                        device_ring_loss,
                        CoreType::WORKER);
                    capability.loss.device_ring = device_ring_loss[0];
                }
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to refresh device loss counters for device {}: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
        capabilities.push_back(capability);
    }
    return capabilities;
}

void RealtimeProfilerManager::publish_pages(
    DeviceState& dev_state,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<tt::ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    auto is_record = [](const uint32_t* page) {
        return (page[2] & 0xFFFF) != 0 && page[3] != REALTIME_PROFILER_SYNC_MARKER_ID;
    };
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double sync_frequency = dev_state.sync_frequency;
    const DataCollector* const data_collector = data_collector_;
    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = page_buf + page * kPageWords;
        if (!is_record(rp)) {
            continue;
        }
        const uint32_t runtime_id = rp[2] & 0xFFFF;
        const uint32_t generation = rp[2] >> 16;
        const uint32_t schema_version = rp[3] & 0xFF;
        const uint32_t record_type = (rp[3] >> 8) & 0xF;
        if (schema_version != REALTIME_PROFILER_RECORD_SCHEMA_VERSION ||
            (record_type != REALTIME_PROFILER_RECORD_TYPE_INTERVAL &&
             record_type != REALTIME_PROFILER_RECORD_TYPE_RESET_OBSERVED)) {
            continue;
        }
        const uint32_t source_loss_low = rp[7];
        if (source_loss_low < dev_state.last_source_loss_low) {
            dev_state.source_loss_high += uint64_t{1} << 32;
        }
        dev_state.last_source_loss_low = source_loss_low;
        const uint64_t cumulative_source_dropped = dev_state.source_loss_high | source_loss_low;
        records.emplace_back(tt::ProgramRealtimeRecord{
            .runtime_id = runtime_id,
            .chip_id = chip_id,
            .start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1],
            .end_timestamp = (static_cast<uint64_t>(rp[4]) << 32) | rp[5],
            .frequency = sync_frequency,
            .kernel_sources = data_collector->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(runtime_id)),
            // The supported realtime-profiler topology has exactly one CQ.
            // Wire word 3 bits 15:12 remain reserved and are not decoded.
            .command_queue_id = 0,
            .dispatch_stream = (rp[3] >> 16) & 0xFF,
            .generation = generation,
            .sequence = rp[6],
            .schema_version = schema_version,
            .record_type = record_type,
            .cumulative_source_dropped = cumulative_source_dropped,
        });
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

    run_init_sync();

    DataCollector* data_collector = MetalContext::instance(context_id_).data_collector().get();
    for (const auto& dev_state : devices_) {
        data_collector->NotifyRealtimeProfilerActivated(dev_state.chip_id);
        data_collector->NotifyRealtimeProfilerCapability({
            .chip_id = dev_state.chip_id,
            .active = true,
            .inactive_reason = experimental::ProgramRealtimeProfilerInactiveReason::None,
        });
    }

    for (auto& dev_state : devices_) {
        dev_state.pending_first_unthrottled_finish_sync = true;
    }

    data_collector->AttachRealtimeProfilerCallbackListener(this);
    data_collector_ = data_collector;

    // Background receiver thread that polls all device sockets round-robin
    receiver_thread_ = std::thread(&RealtimeProfilerManager::run_receiver, this);
}

void RealtimeProfilerManager::initialize_devices(const std::shared_ptr<MeshDevice>& mesh_device) {
    // HAL offsets are the same for all devices (same arch).
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    TT_ASSERT(
        factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
            realtime_profiler_msgs::realtime_profiler_msg_t::Field::loss_descriptor_full) ==
        offsetof(realtime_profiler_msg_t, loss_descriptor_full));
    TT_ASSERT(
        factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
            realtime_profiler_msgs::realtime_profiler_msg_t::Field::loss_device_ring) ==
        offsetof(realtime_profiler_msg_t, loss_device_ring));
    // TODO: When realtime profiler is supported on Quasar, we'll need to pass in the command queue id(s) here.
    const auto& dispatch_mem_map = MetalContext::instance(context_id_).dispatch_mem_map();
    // TODO: When realtime profiler is supported on Quasar, we'll need to pass in the command queue id(s).
    const uint32_t realtime_profiler_base_addr =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG, /*cq_id=*/0);
    // RealtimeProfilerCoreL1 (ring + D2H sender config) sits past the dispatch carve-outs; the core is off the L1 bank
    // table so the allocator never lands here.
    const uint32_t rt_profiler_core_l1_base =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, /*cq_id=*/0);
    const auto rt_profiler_core_l1_addrs = compute_rt_profiler_core_l1_addrs(rt_profiler_core_l1_base);

    // RT_PROFILER_SOCKET_CONFIG_SIZE has headroom over today's SocketSenderSize, but assert
    // it here so a future growth of the sender config triggers a deterministic startup failure.
    TT_FATAL(
        RT_PROFILER_SOCKET_CONFIG_SIZE >= D2HSocket::required_config_buffer_size(),
        "RT_PROFILER_SOCKET_CONFIG_SIZE ({} B) is smaller than D2HSocket's required config "
        "buffer size ({} B). Bump RT_PROFILER_SOCKET_CONFIG_SIZE in "
        "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp and rebuild.",
        RT_PROFILER_SOCKET_CONFIG_SIZE,
        D2HSocket::required_config_buffer_size());
    uint32_t config_buffer_addr_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::config_buffer_addr);
    uint32_t sync_request_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_request);
    uint32_t sync_host_timestamp_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_host_timestamp);
    uint32_t profiler_msg_config_field_addr = realtime_profiler_base_addr + config_buffer_addr_offset;

    auto& dispatch_core_manager = MetalContext::instance(context_id_).get_dispatch_core_manager();
    const std::string realtime_profiler_kernel_path = "tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp";
    const std::string realtime_profiler_push_kernel_path =
        "tt_metal/impl/dispatch/kernels/cq_realtime_profiler_push.cpp";

    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }

        IDevice* device = mesh_device->get_device(coord);
        auto device_id = device->id();
        const uint16_t channel =
            MetalContext::instance(context_id_).get_cluster().get_assigned_channel_for_device(device_id);
        evaluated_chip_ids_.push_back(device_id);

        auto eligibility = dispatch_core_manager.evaluate_realtime_profiler_eligibility(device);
        if (!eligibility.enabled) {
            MetalContext::instance(context_id_)
                .data_collector()
                ->NotifyRealtimeProfilerCapability({
                    .chip_id = static_cast<uint32_t>(device_id),
                    .active = false,
                    .inactive_reason = eligibility.inactive_reason,
                });
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }
        CoreCoord realtime_profiler_core = eligibility.core;

        std::optional<CoreCoord> dispatch_s_core_for_disable;
        if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, channel, 0)) {
            const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, channel, 0);
            dispatch_s_core_for_disable = CoreCoord(dispatch_s_cxy.x, dispatch_s_cxy.y);
        }
        const uint32_t remote_state_addr_field_offset =
            factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_remote_state_addr);
        bool dispatch_activation_published = false;
        ScopeExit disable_unactivated_observer([&] {
            if (dispatch_activation_published || !dispatch_s_core_for_disable.has_value()) {
                return;
            }
            try {
                std::vector<uint32_t> disabled_marker = {REALTIME_PROFILER_REMOTE_STATE_DISABLED};
                tt::tt_metal::detail::WriteToDeviceL1(
                    device,
                    *dispatch_s_core_for_disable,
                    realtime_profiler_base_addr + remote_state_addr_field_offset,
                    disabled_marker,
                    CoreType::WORKER);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "Failed to release unactivated realtime-profiler observer on device {}: {}",
                    device_id,
                    e.what());
            } catch (...) {
                log_warning(
                    tt::LogMetal, "Failed to release unactivated realtime-profiler observer on device {}", device_id);
            }
        });

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Using reserved tensix ({}, {}) for real-time profiler on device {}",
            realtime_profiler_core.x,
            realtime_profiler_core.y,
            device_id);

        DeviceState dev_state;
        dev_state.device = device;
        dev_state.chip_id = device_id;
        dev_state.mesh_coord = coord;
        dev_state.realtime_profiler_core = realtime_profiler_core;
        // Single base past UNRESERVED, sub-addresses via offsetof, bypassing the allocator.
        dev_state.core_l1 = rt_profiler_core_l1_addrs;

        auto sender_core = MeshCoreCoord{coord, realtime_profiler_core};

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Initializing real-time profiler D2H socket for device {} on MeshDevice {}",
            device_id,
            mesh_device->id());

        // D2H socket construction (host pinning / hugepage / UMD) is fragile, so catch and skip this device rather than
        // abort the run.
        try {
            // Pass the carve-out L1 sender-config address so D2HSocket doesn't MeshBuffer::create on a core absent from
            // the L1 bank table.
            dev_state.socket = std::make_unique<D2HSocket>(
                mesh_device,
                sender_core,
                RealtimeProfilerRuntimeSizes::fifo_size,
                D2HSocket::ExternalConfigBuffer{.address = dev_state.core_l1.socket_config},
                D2HSocket::ProcessScope::InProcess);
            dev_state.socket->set_page_size(RealtimeProfilerRuntimeSizes::page_size);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "Real-time profiler disabled on device {}: D2H socket construction failed ({}). "
                "This typically indicates a host-side memory pinning / hugepage mapping issue "
                "(e.g. IOMMU misconfiguration or UMD DMA pin failure). Continuing without RT "
                "profiler on this device.",
                device_id,
                e.what());
            MetalContext::instance(context_id_)
                .data_collector()
                ->NotifyRealtimeProfilerCapability({
                    .chip_id = static_cast<uint32_t>(device_id),
                    .active = false,
                    .inactive_reason = experimental::ProgramRealtimeProfilerInactiveReason::SocketInitializationFailed,
                });
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }

        dev_state.sync_request_addr = realtime_profiler_base_addr + sync_request_offset;
        dev_state.sync_host_ts_addr = realtime_profiler_base_addr + sync_host_timestamp_offset;

        struct DispatchActivation {
            CoreCoord core;
            uint32_t profiler_core_noc_xy;
            uint32_t remote_state_addr;
            uint32_t profiler_core_noc_xy_field_addr;
            uint32_t remote_state_addr_field_addr;
        };
        std::optional<DispatchActivation> dispatch_activation;

        // Prepare dispatch activation, but do not publish it until the reserved-core program is launched.
        if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, channel, 0)) {
            const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, channel, 0);
            CoreCoord dispatch_s_core(dispatch_s_cxy.x, dispatch_s_cxy.y);

            CoreCoord realtime_profiler_virtual =
                device->virtual_core_from_logical_core(realtime_profiler_core, CoreType::WORKER);
            uint32_t realtime_profiler_noc_xy =
                hal.noc_xy_encoding(realtime_profiler_virtual.x, realtime_profiler_virtual.y);

            uint32_t realtime_profiler_core_noc_xy_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_core_noc_xy);
            uint32_t realtime_profiler_state_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_state);
            uint32_t realtime_profiler_core_state_addr = realtime_profiler_base_addr + realtime_profiler_state_offset;
            dispatch_activation = DispatchActivation{
                .core = dispatch_s_core,
                .profiler_core_noc_xy = realtime_profiler_noc_xy,
                .remote_state_addr = realtime_profiler_core_state_addr,
                .profiler_core_noc_xy_field_addr = realtime_profiler_base_addr + realtime_profiler_core_noc_xy_offset,
                .remote_state_addr_field_addr = realtime_profiler_base_addr + remote_state_addr_field_offset,
            };
            dev_state.dispatch_s_core = dispatch_s_core;
            dev_state.dispatch_s_profiler_msg_addr = realtime_profiler_base_addr;
        }

        // dispatch_d, dispatch_s, and TRISC0 publish their owned protocol state
        // before activation. Poll the three contiguous ready words with a finite
        // initialization budget; do not launch the reserved-core program unless
        // the dispatch-side protocol is constructed and version-matched.
        bool dispatch_protocol_ready = false;
        if (dispatch_activation.has_value()) {
            constexpr auto kDispatchProtocolReadyTimeout = std::chrono::seconds(3);
            constexpr auto kDispatchProtocolReadyPoll = std::chrono::milliseconds(1);
            const uint32_t dispatch_d_ready_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                realtime_profiler_msgs::realtime_profiler_msg_t::Field::dispatch_d_ready);
            const uint32_t dispatch_s_ready_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                realtime_profiler_msgs::realtime_profiler_msg_t::Field::dispatch_s_ready);
            const uint32_t observer_ready_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                realtime_profiler_msgs::realtime_profiler_msg_t::Field::observer_ready);
            TT_ASSERT(dispatch_s_ready_offset == dispatch_d_ready_offset + sizeof(uint32_t));
            TT_ASSERT(observer_ready_offset == dispatch_s_ready_offset + sizeof(uint32_t));

            const auto deadline = std::chrono::steady_clock::now() + kDispatchProtocolReadyTimeout;
            std::vector<uint32_t> ready_words(3, 0);
            do {
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device,
                    dispatch_activation->core,
                    realtime_profiler_base_addr + dispatch_d_ready_offset,
                    3 * sizeof(uint32_t),
                    ready_words,
                    CoreType::WORKER);
                dispatch_protocol_ready = ready_words[0] == REALTIME_PROFILER_PROTOCOL_VERSION &&
                                          ready_words[1] == REALTIME_PROFILER_PROTOCOL_VERSION &&
                                          ready_words[2] == REALTIME_PROFILER_PROTOCOL_VERSION;
                if (!dispatch_protocol_ready) {
                    std::this_thread::sleep_for(kDispatchProtocolReadyPoll);
                }
            } while (!dispatch_protocol_ready && std::chrono::steady_clock::now() < deadline);
        }

        if (!dispatch_protocol_ready) {
            log_warning(
                tt::LogMetal,
                "Real-time profiler disabled on device {}: dispatch protocol readiness timed out or version "
                "mismatched",
                device_id);
            MetalContext::instance(context_id_)
                .data_collector()
                ->NotifyRealtimeProfilerCapability({
                    .chip_id = static_cast<uint32_t>(device_id),
                    .active = false,
                    .inactive_reason =
                        experimental::ProgramRealtimeProfilerInactiveReason::ProtocolInitializationFailed,
                });
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }

        // Ring buffer (BRISC->NCRISC handoff) at a fixed carve-out offset; not Buffer::create'd since the core is off
        // the L1 bank table.
        const uint32_t ring_buffer_addr = dev_state.core_l1.ring_buffer;

        // Zero the ring buffer header (everything before RtProfilerRingBuffer::data) to
        // clear stale state from a previous run.
        {
            constexpr uint32_t kRingHeaderBytes = offsetof(RtProfilerRingBuffer, data);
            static_assert(kRingHeaderBytes % sizeof(uint32_t) == 0, "Ring header must be uint32-aligned");
            std::vector<uint32_t> zero_header(kRingHeaderBytes / sizeof(uint32_t), 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, ring_buffer_addr, zero_header, CoreType::WORKER);
        }

        // Zero realtime_profiler_msg_t before launch: stale L1 values misbehave at BRISC/NCRISC boot (garbage socket
        // config, premature sync, phantom marker, corrupt state machine).
        {
            const uint32_t profiler_msg_size = factory.size_of<realtime_profiler_msgs::realtime_profiler_msg_t>();
            const uint32_t profiler_msg_words = profiler_msg_size / sizeof(uint32_t);
            std::vector<uint32_t> zero_msg(profiler_msg_words, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, realtime_profiler_base_addr, zero_msg, CoreType::WORKER);
        }

        // Compile and launch RT-profiler kernels (BRISC reader + NCRISC pusher); Program owned by dev_state so its
        // kernel metadata outlives this scope for tt-inspector.
        {
            dev_state.realtime_profiler_program = std::make_unique<Program>();
            auto& realtime_profiler_program = *dev_state.realtime_profiler_program;

            uint32_t dispatch_core_noc_x = 0;
            uint32_t dispatch_core_noc_y = 0;
            uint32_t dispatch_data_addr_a = 0;
            uint32_t dispatch_data_addr_b = 0;
            if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, channel, 0)) {
                const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, channel, 0);
                CoreCoord dispatch_s_virtual = device->virtual_core_from_logical_core(
                    CoreCoord(dispatch_s_cxy.x, dispatch_s_cxy.y), CoreType::WORKER);
                dispatch_core_noc_x = dispatch_s_virtual.x;
                dispatch_core_noc_y = dispatch_s_virtual.y;

                uint32_t kernel_start_a_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_start_a);
                uint32_t kernel_start_b_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_start_b);
                dispatch_data_addr_a = realtime_profiler_base_addr + kernel_start_a_offset;
                dispatch_data_addr_b = realtime_profiler_base_addr + kernel_start_b_offset;
            }

            DataMovementConfig brisc_config;
            brisc_config.processor = DataMovementProcessor::RISCV_0;
            brisc_config.noc = NOC::RISCV_0_default;
            brisc_config.defines["DISPATCH_CORE_NOC_X"] = std::to_string(dispatch_core_noc_x);
            brisc_config.defines["DISPATCH_CORE_NOC_Y"] = std::to_string(dispatch_core_noc_y);
            brisc_config.defines["DISPATCH_DATA_ADDR_A"] = std::to_string(dispatch_data_addr_a);
            brisc_config.defines["DISPATCH_DATA_ADDR_B"] = std::to_string(dispatch_data_addr_b);
            brisc_config.defines["RING_BUFFER_ADDR"] = std::to_string(ring_buffer_addr);
            brisc_config.defines["REALTIME_PROFILER_MSG_ADDR"] = std::to_string(realtime_profiler_base_addr);
            CreateKernel(
                realtime_profiler_program, realtime_profiler_kernel_path, realtime_profiler_core, brisc_config);

            DataMovementConfig ncrisc_config;
            ncrisc_config.processor = DataMovementProcessor::RISCV_1;
            ncrisc_config.noc = NOC::RISCV_1_default;
            ncrisc_config.defines["RING_BUFFER_ADDR"] = std::to_string(ring_buffer_addr);
            ncrisc_config.defines["REALTIME_PROFILER_MSG_ADDR"] = std::to_string(realtime_profiler_base_addr);
            CreateKernel(
                realtime_profiler_program, realtime_profiler_push_kernel_path, realtime_profiler_core, ncrisc_config);

            tt::tt_metal::detail::CompileProgram(device, realtime_profiler_program, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::detail::WriteRuntimeArgsToDevice(
                device, realtime_profiler_program, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::detail::LaunchProgram(
                device, realtime_profiler_program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

            // realtime_profiler_msg_t is outside mailboxes_t, so LaunchProgram's writes do
            // not race with config_buffer_addr; ordering this write here is intentional.
            uint32_t config_buffer_addr = dev_state.socket->get_config_buffer_address();
            std::vector<uint32_t> addr_data = {config_buffer_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, profiler_msg_config_field_addr, addr_data, CoreType::WORKER);

            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: launched real-time profiler BRISC+NCRISC kernels on core ({}, {}), "
                "ring_buffer_addr=0x{:x}, config_buffer_addr=0x{:x}",
                device_id,
                realtime_profiler_core.x,
                realtime_profiler_core.y,
                ring_buffer_addr,
                config_buffer_addr);
        }

        if (dispatch_activation.has_value()) {
            // Publish the remote address only after the reserved-core program is live, then publish the nonzero NOC
            // coordinate as the activation release observed by dispatch_s and TRISC0.
            std::vector<uint32_t> remote_state_addr_data = {dispatch_activation->remote_state_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_activation->core,
                dispatch_activation->remote_state_addr_field_addr,
                remote_state_addr_data,
                CoreType::WORKER);
            std::vector<uint32_t> noc_xy_data = {dispatch_activation->profiler_core_noc_xy};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_activation->core,
                dispatch_activation->profiler_core_noc_xy_field_addr,
                noc_xy_data,
                CoreType::WORKER);
            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: activated dispatch_s ({}, {}) with profiler noc_xy=0x{:x}, "
                "remote_state_addr=0x{:x}",
                device_id,
                dispatch_activation->core.x,
                dispatch_activation->core.y,
                dispatch_activation->profiler_core_noc_xy,
                dispatch_activation->remote_state_addr);
            dispatch_activation_published = true;
        }

        disable_unactivated_observer.release();

        MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
        devices_.push_back(std::move(dev_state));
    }
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
            "[Real-time profiler] Device {} D2H FIFO reached capacity ({} pages); profiler data may be dropped",
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

void RealtimeProfilerManager::run_consumer(Consumer& consumer) {
    tracy::SetThreadName(fmt::format("RtProfilerConsumer{}", consumer.handle).c_str());
    std::vector<tt::ProgramRealtimeRecord> records(
        std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * devices_.size()));
    uint64_t reported_dropped = 0;

    auto deliver_batch = [&](std::span<const tt::ProgramRealtimeRecord> batch, uint64_t dropped_total) {
        TTZoneScopedDNC(RT_PROFILER, "Callback", 0xF032E6);
        TTZoneValueD(RT_PROFILER, batch.size());
        const uint64_t dropped_delta = dropped_total - reported_dropped;
        if (TTZoneIsActiveD(RT_PROFILER) && dropped_delta > 0) {
            const auto dropped_txt = fmt::format("dropped {}", dropped_delta);
            TTZoneTextD(RT_PROFILER, dropped_txt.c_str(), dropped_txt.size());
        }
        std::vector<tt::ProgramRealtimeProfilerDeviceLossSnapshot> device_loss;
        device_loss.reserve(devices_.size());
        for (const auto& record : batch) {
            auto snapshot = std::find_if(device_loss.begin(), device_loss.end(), [&](const auto& entry) {
                return entry.chip_id == record.chip_id;
            });
            if (snapshot == device_loss.end()) {
                device_loss.push_back({record.chip_id, record.cumulative_source_dropped});
            } else {
                snapshot->cumulative_source_dropped = record.cumulative_source_dropped;
            }
        }
        const tt::ProgramRealtimeRecordBatch arg{batch, dropped_delta, device_loss};
        reported_dropped = dropped_total;
        try {
            consumer.callback(arg);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an exception: {}", e.what());
        } catch (...) {
            log_warning(tt::LogMetal, "[Real-time profiler] Callback threw an unknown exception");
        }
    };

    while (true) {
        // stop_consumer sets the stop mode then wakes, so waiting on a token sampled only inside wait() could miss that
        // wake and hang
        const auto token = consumer.reader.wait_token();
        std::span<tt::ProgramRealtimeRecord> batch = consumer.reader.read_batch(records);
        const uint64_t dropped_total = consumer.reader.dropped();
        const ConsumerStopMode stop_mode = consumer.stop_mode.load(std::memory_order_acquire);
        if (stop_mode == ConsumerStopMode::StopWithoutDrain) {
            break;
        }
        if (!batch.empty()) {
            deliver_batch(batch, dropped_total);
        } else if (stop_mode == ConsumerStopMode::DrainThenStop) {
            break;
        } else {
            TTZoneScopedDN(RT_PROFILER, "Wait");
            consumer.reader.wait(token);
        }
    }
    consumer.dropped = consumer.reader.dropped();
}

void RealtimeProfilerManager::stop_consumer(Consumer& consumer, ConsumerStopMode stop_mode) {
    consumer.stop_mode.store(stop_mode, std::memory_order_release);
    ring_->writer().wake_readers();
    if (consumer.thread.joinable()) {
        consumer.thread.join();
    }
}

void RealtimeProfilerManager::on_callback_registered(
    tt::ProgramRealtimeProfilerCallbackHandle handle, const tt::ProgramRealtimeProfilerCallback& callback) {
    auto consumer = std::make_unique<Consumer>(ring_->make_reader(), callback, handle);
    Consumer* raw = consumer.get();
    std::lock_guard<std::mutex> lock(consumers_mutex_);
    const auto caller = std::this_thread::get_id();
    const bool from_callback_thread =
        std::ranges::any_of(consumers_, [caller](const auto& kv) { return kv.second->thread.get_id() == caller; });
    TT_FATAL(!from_callback_thread, "A real-time profiler callback must not register callbacks");
    consumers_.emplace(handle, std::move(consumer));
    raw->thread = std::thread([this, raw]() { run_consumer(*raw); });
}

void RealtimeProfilerManager::on_callback_unregistered(tt::ProgramRealtimeProfilerCallbackHandle handle) {
    std::unique_ptr<Consumer> consumer;
    {
        std::lock_guard<std::mutex> lock(consumers_mutex_);
        auto it = consumers_.find(handle);
        if (it == consumers_.end()) {
            return;
        }
        const auto caller = std::this_thread::get_id();
        const bool from_callback_thread =
            std::ranges::any_of(consumers_, [caller](const auto& kv) { return kv.second->thread.get_id() == caller; });
        TT_FATAL(!from_callback_thread, "A real-time profiler callback must not unregister callbacks");
        consumer = std::move(it->second);
        consumers_.erase(it);
    }
    stop_consumer(*consumer, ConsumerStopMode::StopWithoutDrain);
    const uint64_t dropped = consumer->dropped;
    if (dropped > 0) {
        log_warning(tt::LogMetal, "[Real-time profiler] Callback {} dropped {} record(s)", handle, dropped);
    }
}

RealtimeProfilerManager::~RealtimeProfilerManager() { shutdown(); }

void RealtimeProfilerManager::shutdown() {
    constexpr auto kShutdownKernelExitGrace = std::chrono::milliseconds(100);
    MetalContext::instance(context_id_).data_collector()->DetachRealtimeProfilerCallbackListener(this);

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
    if (!devices_.empty()) {
        std::this_thread::sleep_for(kShutdownKernelExitGrace);
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

    // dispatch_s has completed its bounded observer stop/drain before the CQ
    // terminates. Preserve that terminal diagnostic snapshot after close so a
    // caller can audit losses that necessarily occur too late for an ordinary
    // in-flight callback record. A later evaluation of the same chip replaces
    // this entry.
    auto* context_data_collector = MetalContext::instance(context_id_).data_collector().get();
    for (auto capability : get_device_capabilities()) {
        capability.active = false;
        context_data_collector->NotifyRealtimeProfilerCapability(capability);
        if (capability.loss.terminal_descriptor != 0 || capability.loss.terminal_record != 0 ||
            capability.loss.observer_stop_timeout != 0) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} shutdown losses: terminal_descriptor={}, terminal_record={}, "
                "observer_stop_timeout={}",
                capability.chip_id,
                capability.loss.terminal_descriptor,
                capability.loss.terminal_record,
                capability.loss.observer_stop_timeout);
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
        context_data_collector->NotifyRealtimeProfilerDeactivated(dev_state.chip_id);
    }
    evaluated_chip_ids_.clear();
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
