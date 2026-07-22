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

#include <enchantum/enchantum.hpp>
#include <fmt/core.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tt_align.hpp>
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
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/dispatch/realtime_profiler_tracy_handler.hpp"

namespace tt::tt_metal::distributed {

namespace {

// Minimum wall time between full init calibrations (run_sync + constructor SYNC_CHECK) and
// between finish-path sync checks, per physical chip. Matches the finish-path throttle.
constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

constexpr auto kFinishSyncRequestDelay = std::chrono::milliseconds(5);
constexpr auto kFinishSyncResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kSyncResponsePollBackoff = std::chrono::microseconds(100);

// Last time we completed a full init sync (run_sync success) for a chip, process-wide
// (across MeshDevice open/close). Used to avoid repeating ~0.5s+ run_sync on every mesh
// open when the same host chips are frequently reconstructed.
std::mutex g_rt_profiler_init_sync_mu;
std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> g_rt_profiler_last_init_sync_by_chip;

// Sync marker ID — must match device-side REALTIME_PROFILER_SYNC_MARKER_ID.
constexpr uint32_t REALTIME_PROFILER_SYNC_MARKER_ID = 0xFFFFFFFF;

// Real-time profiler runtime constants. On-device L1 layout sizes are reused from
// realtime_profiler_ring_buffer.hpp so host and device share a single source of truth.
struct RealtimeProfilerRuntimeSizes {
    static constexpr uint32_t fifo_pages = 32768;                  // host D2H FIFO depth, in pages
    static constexpr uint32_t page_size = RT_PROFILER_ENTRY_SIZE;  // host page size == ring entry size
    static constexpr uint32_t fifo_size = fifo_pages * page_size;  // pinned-host FIFO, in bytes (2 MiB)
    static constexpr uint32_t core_l1_size = sizeof(RealtimeProfilerCoreL1);
};

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// Compute the RT-profiler tensix L1 carve-out addresses for a given RealtimeProfilerCoreL1
// base, anchored past dispatch_mem_map's UNRESERVED so the layout sits outside the
// user-space allocator.
inline RealtimeProfilerCoreL1Addrs compute_rt_profiler_core_l1_addrs(uint32_t base) {
    return {
        .base = base,
        .ring_buffer = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, ring)),
        .socket_config = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, socket_config)),
    };
}

// Result of evaluating whether the real-time profiler can be brought up on a device.
struct RealtimeProfilerEligibility {
    bool enabled = false;
    CoreCoord core;  // Only meaningful when enabled == true.
};

// Consolidated eligibility check; logs the reason for disabling and returns
// {enabled=false} on failure.
//
// Evaluates against the device's owning context (passed in as `context_id`) rather than
// bare MetalContext::instance(): the latter would route through the inline non-default
// fallback in instance() and pick whichever context happens to populate the global
// lookup first. In silicon-first coexistence (#38445), that fallback returns the silicon
// DEFAULT_CONTEXT_ID even when `device` is a mock device, falsely enabling the profiler
// on mock and SEGV'ing in LaunchProgram. See #39849.
//
// Checks (in order):
//   0. Target is not mock or emulated (extends #43968's Mock-only short-circuit to also
//      cover Emule; D2HSocket requires a real PCIe hugepage in either case).
//   1. Device is MMIO-capable (D2H sockets need a PCIe-connected sender core).
//   2. D2H socket memory-allocation path is supported (64-bit PCIe addressing requires IOMMU).
//   3. Fabric tensix datamover (MUX / UDM) is disabled (it competes for the same dispatch pool).
//   4. A tensix core was reserved for the RT profiler at dispatch_core_manager construction.
//   5. Reserved coordinate lives inside the logical TENSIX grid.
//   6. Kernels are not nullified (DEBUG_NULL_KERNELS / TT_METAL_NULL_KERNELS).
//   7. Reserved profiler core's L1 bank fits the ring + socket-config layout.
RealtimeProfilerEligibility evaluate_realtime_profiler_eligibility(IDevice* device, ContextId context_id) {
    auto device_id = device->id();
    auto& metal = MetalContext::instance(context_id);
    const auto& hal = metal.hal();
    const auto& cluster = metal.get_cluster();
    auto& dispatch_core_manager = metal.get_dispatch_core_manager();

    // Subsumes the Mock-only short-circuit added in #43968: is_mock_or_emulated() also
    // catches Emule, and is the canonical accessor used throughout metal_context.cpp /
    // device.cpp. D2HSocket::init_host_buffer_hugepage dereferences a real PCIe hugepage
    // and faults on either target, so gate here before any per-device profiler state is
    // constructed.
    if (cluster.is_mock_or_emulated()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: target is mock or emulated; D2H sockets "
            "require a real PCIe hugepage that is not present in mock/emulated flows.",
            device_id);
        return {};
    }

    // ttsim: the simulator's D2H socket exists but its device kernels run many orders of
    // magnitude slower than real silicon, so the 2 s WriteToDeviceL1/sync poll deadline
    // in run_sync() always trips before the profiler core can respond. That burns ~30 s
    // per chip during MeshDevice bring-up, and on WH (where the 64-bit-PCIe gate below
    // does NOT fire) it deadlocks downstream waiters that depend on first_unthrottled
    // finish_sync. Skip the profiler entirely on Simulator targets; performance traces
    // are not interesting on the sim anyway.
    if (cluster.get_target_device_type() == tt::TargetDevice::Simulator) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: target is Simulator; D2H sync polls "
            "cannot meet real-time deadlines against ttsim's emulated PCIe.",
            device_id);
        return {};
    }

    if (!device->is_mmio_capable()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: device is not MMIO-capable (remote device). "
            "D2H sockets require the sender core to sit on a PCIe-connected chip.",
            device_id);
        return {};
    }

    if (hal.get_supports_64_bit_pcie_addressing() && !cluster.is_iommu_enabled()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: this architecture uses 64-bit PCIe "
            "addressing for the D2H socket, which requires IOMMU to be enabled on the host. "
            "IOMMU is currently disabled and no hugepage fallback is available. Enable IOMMU "
            "(or run on a system that has it) to re-enable RT profiler.",
            device_id);
        return {};
    }

    const auto fabric_tensix_config = metal.get_fabric_tensix_config();
    if (fabric_tensix_config != tt_fabric::FabricTensixConfig::DISABLED) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: fabric tensix datamover is enabled "
            "(FabricTensixConfig={}, FabricUDMMode={}), and fabric_mux_core() will drain the "
            "remaining dispatch-pool cores at fabric-init time. Reserving a tensix for the RT "
            "profiler on top of that tips the pool into exhaustion on small-pool chips. "
            "Disable the fabric tensix datamover to re-enable RT profiler.",
            device_id,
            enchantum::to_string(fabric_tensix_config),
            enchantum::to_string(metal.get_fabric_udm_mode()));
        return {};
    }

    std::optional<tt_cxy_pair> reserved = dispatch_core_manager.get_reserved_realtime_profiler_core(device_id);
    if (!reserved.has_value()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: no tensix core could be reserved for the "
            "RT profiler. Dispatch is configured for ETH cores, which cannot run the RT profiler "
            "BRISC kernel. Switch to DispatchCoreConfig(DispatchCoreType::WORKER) to re-enable RT "
            "profiler.",
            device_id);
        return {};
    }

    CoreCoord core(reserved->x, reserved->y);

    const auto& soc = cluster.get_soc_desc(device_id);
    CoreCoord tensix_grid = soc.get_grid_size(CoreType::TENSIX);
    if (core.x >= tensix_grid.x || core.y >= tensix_grid.y) {
        log_warning(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: reserved core ({}, {}) is outside the "
            "TENSIX logical grid ({}, {}).",
            device_id,
            core.x,
            core.y,
            tensix_grid.x,
            tensix_grid.y);
        return {};
    }

    if (metal.rtoptions().get_kernels_nullified()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: null-kernels mode is active "
            "(TT_METAL_NULL_KERNELS / set_kernels_nullified). The RT profiler kernel "
            "would be replaced with a stub and could not respond to host syncs, and "
            "there are no real user kernels to profile in this mode.",
            device_id);
        return {};
    }

    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t core_l1_size_aligned = tt::align(RealtimeProfilerRuntimeSizes::core_l1_size, l1_alignment);
    const DeviceAddr l1_bank_size = device->allocator()->get_bank_size(BufferType::L1);
    if (l1_bank_size < core_l1_size_aligned) {
        log_warning(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: not enough user-allocatable L1 on the "
            "reserved profiler core ({}, {}) for the RT-profiler L1 layout "
            "(need {} B, L1 bank size is {} B). Increase worker_l1_size by at least {} B "
            "(or leave it at the default) to re-enable RT profiler.",
            device_id,
            core.x,
            core.y,
            core_l1_size_aligned,
            l1_bank_size,
            core_l1_size_aligned - l1_bank_size);
        return {};
    }

    return {.enabled = true, .core = core};
}

// Host clock wrapper for the RT profiler sync handshake. Tracy stubs TracyGetCpuTime() /
// TracyGetTimerMul() to 0 when disabled, which would write sync_host_timestamp = 0 to L1
// and stall the device handshake. Fall back to steady_clock in that case.
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
    const unsigned hc = std::thread::hardware_concurrency();
    const size_t worker_count = std::min(indices.size(), static_cast<size_t>(std::max(1u, hc)));
    if (worker_count <= 1) {
        for (size_t di : indices) {
            callable(di);
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
                callable(indices[k]);
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

void RealtimeProfilerManager::publish_pages(
    const DeviceState& dev_state,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<tt::ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    auto is_record = [](const uint32_t* page) { return page[2] != 0 && page[3] != REALTIME_PROFILER_SYNC_MARKER_ID; };
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double sync_frequency = dev_state.sync_frequency;
    const DataCollector* const data_collector = data_collector_;
    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = page_buf + page * kPageWords;
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
        tt::NotifyProgramRealtimeProfilerActivated(dev_state.chip_id);
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

void RealtimeProfilerManager::initialize_devices(const std::shared_ptr<MeshDevice>& mesh_device) {
    // HAL offsets are the same for all devices (same arch).
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    // realtime_profiler_msg_t lives in a dispatch-core-local L1 region assigned by
    // CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG (only reachable on dispatch cores
    // and the reserved RT-profiler tensix).
    const auto& dispatch_mem_map = MetalContext::instance(context_id_).dispatch_mem_map();
    const uint32_t realtime_profiler_base_addr =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG);
    // RealtimeProfilerCoreL1 (ring + D2H socket sender config) sits past the dispatch
    // carve-outs on the reserved profiler tensix; the core is excluded from the L1 bank
    // table so the user-space allocator can never land here.
    const uint32_t rt_profiler_core_l1_base =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED);
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

        auto eligibility = evaluate_realtime_profiler_eligibility(device, context_id_);
        if (!eligibility.enabled) {
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }
        CoreCoord realtime_profiler_core = eligibility.core;

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
        // Single base anchored past dispatch_mem_map's UNRESERVED, with all sub-addresses
        // derived via offsetof — bypasses the user-space allocator entirely.
        dev_state.core_l1 = rt_profiler_core_l1_addrs;

        auto sender_core = MeshCoreCoord{coord, realtime_profiler_core};

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Initializing real-time profiler D2H socket for device {} on MeshDevice {}",
            device_id,
            mesh_device->id());

        // Defensive: the eligibility gate above filters known-bad configurations, but D2H
        // socket construction (host pinning / hugepage / UMD interaction) has been
        // historically fragile, so we catch and skip this device on failure rather than
        // abort the run.
        try {
            // Pass the L1 sender-config address from the dispatch carve-out so D2HSocket
            // does not allocate via MeshBuffer::create on a reserved dispatch core (which
            // would crash get_buffer_pages on cores not in the L1 bank table).
            dev_state.socket = std::make_unique<D2HSocket>(
                mesh_device,
                sender_core,
                RealtimeProfilerRuntimeSizes::fifo_size,
                D2HSocket::ExternalConfigBuffer{.address = dev_state.core_l1.socket_config});
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
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }

        dev_state.sync_request_addr = realtime_profiler_base_addr + sync_request_offset;
        dev_state.sync_host_ts_addr = realtime_profiler_base_addr + sync_host_timestamp_offset;

        // Write real-time profiler core info into the dispatch carve-out for termination signaling.
        if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
            const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, 0, 0);
            CoreCoord dispatch_s_core(dispatch_s_cxy.x, dispatch_s_cxy.y);

            CoreCoord realtime_profiler_virtual =
                device->virtual_core_from_logical_core(realtime_profiler_core, CoreType::WORKER);
            uint32_t realtime_profiler_noc_xy =
                hal.noc_xy_encoding(realtime_profiler_virtual.x, realtime_profiler_virtual.y);

            uint32_t realtime_profiler_core_noc_xy_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_core_noc_xy);
            uint32_t remote_state_addr_field_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_remote_state_addr);
            uint32_t realtime_profiler_state_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_state);
            uint32_t realtime_profiler_core_state_addr = realtime_profiler_base_addr + realtime_profiler_state_offset;

            std::vector<uint32_t> noc_xy_data = {realtime_profiler_noc_xy};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                realtime_profiler_base_addr + realtime_profiler_core_noc_xy_offset,
                noc_xy_data,
                CoreType::WORKER);

            std::vector<uint32_t> remote_state_addr_data = {realtime_profiler_core_state_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                realtime_profiler_base_addr + remote_state_addr_field_offset,
                remote_state_addr_data,
                CoreType::WORKER);

            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: wrote real-time profiler core info (noc_xy=0x{:x}, "
                "remote_state_addr=0x{:x}) "
                "to dispatch_s ({}, {})",
                device_id,
                realtime_profiler_noc_xy,
                realtime_profiler_core_state_addr,
                dispatch_s_core.x,
                dispatch_s_core.y);
        }

        // Ring buffer (BRISC->NCRISC handoff) sits at a fixed offset inside the carve-out;
        // not allocated via Buffer::create because the profiler core is excluded from the
        // L1 bank table.
        const uint32_t ring_buffer_addr = dev_state.core_l1.ring_buffer;

        // Get PCIe core NOC-0 coordinates for WH (NCRISC kernel translates to NOC 1).
        uint32_t pcie_noc_x = 0;
        uint32_t pcie_noc_y = 0;
        bool need_pcie_noc_defines = false;
        {
            const auto& cluster = MetalContext::instance(context_id_).get_cluster();
            auto arch = MetalContext::instance(context_id_).hal().get_arch();
            if (arch == tt::ARCH::WORMHOLE_B0) {
                ChipId mmio_device_id = cluster.get_associated_mmio_device(device_id);
                const auto& soc = cluster.get_soc_desc(mmio_device_id);
                const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
                TT_ASSERT(!pcie_cores.empty());
                pcie_noc_x = pcie_cores.front().x;
                pcie_noc_y = pcie_cores.front().y;
                need_pcie_noc_defines = true;
            }
        }
        // Zero the ring buffer header (everything before RtProfilerRingBuffer::data) to
        // clear stale state from a previous run.
        {
            constexpr uint32_t kRingHeaderBytes = offsetof(RtProfilerRingBuffer, data);
            static_assert(kRingHeaderBytes % sizeof(uint32_t) == 0, "Ring header must be uint32-aligned");
            std::vector<uint32_t> zero_header(kRingHeaderBytes / sizeof(uint32_t), 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, ring_buffer_addr, zero_header, CoreType::WORKER);
        }

        // Zero the realtime_profiler_msg_t region before launching the kernels — L1 is not
        // guaranteed zero between runs and stale values misbehave at BRISC/NCRISC boot:
        //   * config_buffer_addr != 0  -> NCRISC reads garbage socket config.
        //   * sync_request != 0        -> BRISC enters sync before the host is ready.
        //   * sync_host_timestamp != 0 -> phantom sync marker pushed on first boot.
        //   * realtime_profiler_state / program_id_fifo_{start,end} corrupt state machine.
        {
            const uint32_t profiler_msg_size = factory.size_of<realtime_profiler_msgs::realtime_profiler_msg_t>();
            const uint32_t profiler_msg_words = profiler_msg_size / sizeof(uint32_t);
            std::vector<uint32_t> zero_msg(profiler_msg_words, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, realtime_profiler_base_addr, zero_msg, CoreType::WORKER);
        }

        // Compile and launch real-time profiler kernels (BRISC reader + NCRISC pusher).
        // The Program is owned by dev_state so it (and its kernel metadata) outlives this
        // scope; otherwise tt-inspector loses track of the running RT-profiler kernels.
        {
            dev_state.realtime_profiler_program = std::make_unique<Program>();
            auto& realtime_profiler_program = *dev_state.realtime_profiler_program;

            uint32_t dispatch_core_noc_x = 0;
            uint32_t dispatch_core_noc_y = 0;
            uint32_t dispatch_data_addr_a = 0;
            uint32_t dispatch_data_addr_b = 0;
            if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
                const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, 0, 0);
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
            if (need_pcie_noc_defines) {
                ncrisc_config.defines["RT_PROFILER_PCIE_NOC_X"] = std::to_string(pcie_noc_x);
                ncrisc_config.defines["RT_PROFILER_PCIE_NOC_Y"] = std::to_string(pcie_noc_y);
            }
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

    // Run our own host-device sync; the device profiler's SyncInfo masks the high word to
    // 12 bits and would shift RT zones by hours in Tracy. Skip full calibration for chips
    // that were init-synced recently (same window as finish-path trigger_sync_check).
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
    }

    // Emit sync verification markers: take one independent device measurement per device
    // and push paired host + device events. In Tracy, the horizontal distance between the
    // host "SYNC_CHECK" zone and the device "SYNC_CHECK" zone is the sync error.
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

        // Same anchor convention as trigger_sync_check: capture host TSC, emit Tracy
        // message, then PCIe write; CalibrateDevice must run before PushSyncCheckMarker
        // or extrapolation skew can exceed the ±10µs test bound.
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
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;
    while (!stop_.load(std::memory_order_acquire)) {
        const bool scan_sync_marker = finish_sync_busy_.load(std::memory_order_acquire);
        const uint32_t num_pages = drain_all_devices(scan_sync_marker, page_buf, record_buf);
        num_pages_received += num_pages;
        const bool sync_requested = finish_sync_requested_.load(std::memory_order_acquire);
        if (scan_sync_marker || sync_requested) {
            service_finish_sync(std::chrono::steady_clock::now(), sync_requested);
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
        const tt::ProgramRealtimeRecordBatch arg{batch, dropped_total - reported_dropped};
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
        const auto batch = consumer.reader.read_batch(records);
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

    // Re-write ring_buffer->terminate as a safety net (dispatch_s already set it via the
    // profiler core's TERMINATE), then give the push kernel time to deliver the last PCIe page.
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
        tt::NotifyProgramRealtimeProfilerDeactivated(dev_state.chip_id);
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

    // Discard pre-existing pages before entering sync mode without reading the data
    // region (its PCIe-mapped bytes can be undefined on the first sync of a fresh
    // MeshDevice). discard_pending_pages() rebases bytes_acked -> bytes_sent and
    // notifies the device.
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

    // Centered linear regression for slope = frequency * tracy_ratio (device cycles per
    // TSC tick). Centering on the mean avoids catastrophic cancellation in the normal
    // equations at the ~10^25 operand magnitudes seen for absolute timestamps.
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

        log_info(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "device_time_at_sync={} cycles",
            dev_state.chip_id,
            samples.size(),
            dev_state.sync_frequency,
            dev_state.first_timestamp);
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
