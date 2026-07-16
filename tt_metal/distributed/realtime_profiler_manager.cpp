// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/realtime_profiler_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

#include <umd/device/warm_reset.hpp>
#include <tt-metalium/experimental/realtime_profiler_packets.hpp>
#include "hostdevcommon/profiler_common.h"
#include "jit_build/build_env_manager.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"
#include "tools/profiler/x280_driver.hpp"

namespace tt::tt_metal::distributed {

namespace {

// Broadcast PROFILER_TERMINATE=1 to every worker core on a device (FULL logical grid, so DISPATCH
// cores are covered too). Makes each producing RISC's ring_ensure_room stop blocking on a full SPSC
// ring and drop markers instead. Needed when the X280 drainer fails to boot: with no drainer the
// rings are never emptied, so without this a producing RISC blocks forever once its ring fills —
// deadlocking the workload (the command-queue completion never arrives). Best-effort; logs and
// continues on write failure.
void broadcast_profiler_terminate(Cluster& cluster, ChipId chip_id, IDevice* device, uint64_t prof_l1) {
    if (device == nullptr) {
        return;
    }
    const uint64_t terminate_addr =
        prof_l1 + static_cast<uint64_t>(kernel_profiler::PROFILER_TERMINATE) * sizeof(uint32_t);
    const uint32_t one = 1;
    const CoreCoord grid = device->logical_grid_size();
    for (uint32_t ly = 0; ly < static_cast<uint32_t>(grid.y); ly++) {
        for (uint32_t lx = 0; lx < static_cast<uint32_t>(grid.x); lx++) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, CoreCoord{lx, ly}, CoreType::WORKER);
            cluster.write_core(&one, sizeof(one), tt_cxy_pair(chip_id, v), terminate_addr);
        }
    }
}

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
    // PROTOTYPING: program-record publishing is DISABLED. The ring is repurposed to carry X280
    // device-zone markers (published by drain_x280_device). The program D2H socket is still drained
    // (to keep its FIFO from filling / sync markers scanned) but its records are dropped here.
    (void)dev_state;
    (void)page_buf;
    (void)num_pages;
    (void)records;
    return;
#if 0
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
#endif
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
    // PROTO: the ring now carries X280 device MARKERS (hundreds of k per run), not the far smaller
    // program-record volume. The default (max_consumer_batch_records*headroom = 131072) is too small
    // -> the lossy BroadcastRing overwrites unread markers, splitting START/END pairs -> orphaned zones.
    // Size it to hold the full run's backlog so shutdown's DrainThenStop pushes everything (lossless).
    // kMaxRingCapacity (4M records ~= 128 MB) covers the observed ~870k markers with headroom.
    // (An earlier "4M crashes" note was a misdiagnosis -- the crash was a dangling x280_dev_ / the
    //  find()-%0 SIGFPE, unrelated to ring size; fixed separately.)
    (void)max_consumer_batch_records;
    ring_.emplace(kMaxRingCapacity);

    // Announce activation; paired with NotifyProgramRealtimeProfilerDeactivated on shutdown.
    // Additionally, for chips where the X280 drainer actually booted (x280_active), announce the
    // "X280 won" signal so the standard DeviceProfiler stands down and doesn't read those chips'
    // SPSC rings (two consumers corrupt the ring head).
    for (const auto& dev_state : devices_) {
        tt::NotifyProgramRealtimeProfilerActivated(dev_state.chip_id);
        if (dev_state.x280_active) {
            tt::NotifyProgramX280ProfilerActivated(dev_state.chip_id);
        }
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

        // Optionally boot the X280 (L2CPU) kernel-zone drainer. Best-effort: on any failure it
        // leaves x280_active=false and the device runs with the standard program-record path only.
        boot_x280_drainer(mesh_device, coord, dev_state);

        MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
        devices_.push_back(std::move(dev_state));
    }

    // Point x280_dev_ at the X280-active element INSIDE devices_, now that the vector is fully built
    // and will not reallocate for the rest of the run. Doing this per-iteration (or inside
    // boot_x280_drainer) is a use-after-free: each push_back can reallocate the vector, and the value
    // boot_x280_drainer saw was a soon-moved-from loop local. run_consumer dereferences x280_dev_ from
    // another thread, so a stale pointer here is the SIGFPE/SIGSEGV in the WorkerZone enrichment path.
    for (auto& dev_state : devices_) {
        if (dev_state.x280_active) {
            x280_dev_ = &dev_state;
            break;
        }
    }
}

void RealtimeProfilerManager::boot_x280_drainer(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& coord, DeviceState& dev_state) {
    const auto device_id = dev_state.chip_id;
    const auto& hal = MetalContext::instance(context_id_).hal();

    // --- Optional: boot the X280 (L2CPU) kernel-zone drainer on this device ---
    // The X280 is the sole consumer of the per-RISC SPSC zone rings; without it a profiler
    // run can deadlock once a ring fills. It drains those rings, PAIRS start/end markers per
    // (core,risc) on-device, and pushes complete device-zone pages through its OWN D2H socket,
    // which the receiver polls alongside the program-record socket. Best-effort: any failure
    // leaves x280_active=false and the device runs without kernel-zone capture.
    try {
        auto& x280_cluster = MetalContext::instance(context_id_).get_cluster();
        const auto& soc = x280_cluster.get_soc_desc(device_id);
        std::string x280_fw = BuildEnvManager::get_instance(context_id_).get_x280_firmware_path(device_id);
        if (x280_cluster.arch() == tt::ARCH::BLACKHOLE && !soc.get_cores(CoreType::L2CPU, CoordSystem::NOC0).empty() &&
            !x280_fw.empty()) {
            constexpr int kL2CpuIndex = 0;  // tile (8,3) — proven single-chip path
            constexpr int kX280PllMhz = 1000;
            // X280 LIM socket md: parked in the gap between MIRRORCTL (0x08018000..~0x0801B000, sized
            // by the drain list) and SINGLECTL (0x0801C000). See profzone.c D8 LIM map.
            constexpr uint32_t kX280ConfigAddr = 0x0801B000u;
            constexpr uint64_t kX280MboxParams = 0x08011000ull;
            constexpr uint64_t kX280MboxResults = 0x08011040ull;
            constexpr uint64_t kX280MboxCoords = 0x08011200ull;
            // D2H FIFO depth (host pinned). At 4 KB (=64 x 64B markers) the X280 filled it in ~0.3ms
            // and then spun on reserve-stalls waiting for the host to free space, throttling its ring
            // drain and back-pressuring the compute cores. Size it to buffer the X280<->host round-trip
            // (1 MB = 16384 markers) so the X280 pipelines ahead instead of stalling.
            constexpr uint32_t kX280Fifo = 1u << 20;  // 1 MiB
            constexpr uint32_t kX280PageSize = 64;

            std::ifstream f(x280_fw, std::ios::binary);
            std::vector<uint8_t> bin((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            while (bin.size() % 4 != 0) {
                bin.push_back(0);
            }
            TT_FATAL(!bin.empty(), "X280 drainer firmware {} is empty", x280_fw);

            const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
            CoreCoord grid = mesh_device->compute_with_storage_grid_size();
            const uint32_t gx = static_cast<uint32_t>(grid.x), gy = static_cast<uint32_t>(grid.y);
            const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;

            // Virtual coords of every logical worker core (what the X280's NOC addresses), and
            // pre-zero each core's SPSC control vector so head/tail start clean.
            std::vector<uint8_t> coord_buf(num_cores * 8, 0);
            std::vector<uint8_t> zero_ctrl(profiler::X280_PROF_CTRL_WORDS * 4, 0);
            for (uint32_t ly = 0; ly < gy; ly++) {
                for (uint32_t lx = 0; lx < gx; lx++) {
                    uint32_t idx = ly * gx + lx;
                    CoreCoord v = x280_cluster.get_virtual_coordinate_from_logical_coordinates(
                        device_id, CoreCoord{lx, ly}, CoreType::WORKER);
                    uint32_t vx = static_cast<uint32_t>(v.x), vy = static_cast<uint32_t>(v.y);
                    std::memcpy(coord_buf.data() + idx * 8 + 0, &vx, 4);
                    std::memcpy(coord_buf.data() + idx * 8 + 4, &vy, 4);
                    x280_cluster.write_core(
                        zero_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, v), prof_l1);
                    // Map the virtual coord the X280 relays back to the NOC0 coord the standard
                    // DeviceProfiler / Tracy use, so kernel-zone lanes line up 1:1 with the DRAM
                    // push profiler's view.
                    const CoreCoord noc0 = x280_cluster.get_physical_coordinate_from_logical_coordinates(
                        device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
                    dev_state.x280_virt_to_noc0[(static_cast<uint64_t>(vx) << 32) | vy] = {
                        static_cast<uint32_t>(noc0.x), static_cast<uint32_t>(noc0.y)};
                    if (lx == 0 && ly == 0) {
                        log_debug(
                            tt::LogMetal,
                            "[Real-time profiler] X280 coord map sample: logical(0,0) virtual=({},{}) "
                            "noc0=({},{})",
                            vx,
                            vy,
                            noc0.x,
                            noc0.y);
                    }
                }
            }

            // PCIe tile (TRANSLATED) the X280 writes its socket pages through.
            const auto pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
            TT_FATAL(!pcie_cores.empty(), "X280 drainer: no PCIe core on device {}", device_id);
            const auto pc = pcie_cores.front();

            dev_state.x280_driver =
                std::make_unique<profiler::X280Driver>(x280_cluster, static_cast<int>(device_id), kL2CpuIndex);
            auto& zx = *dev_state.x280_driver;
            zx.assert_reset();
            zx.load_lim(bin);
            zx.write_block(coord_buf.data(), (uint32_t)coord_buf.size(), kX280MboxCoords);

            // The socket's sender is the X280 L2CPU; its sender_socket_md lands in the X280 LIM.
            const CoreCoord l2phys = profiler::x280_l2cpu_tile(kL2CpuIndex);
            dev_state.x280_socket = std::make_unique<D2HSocket>(
                mesh_device,
                MeshCoreCoord{coord, l2phys},
                kX280Fifo,
                D2HSocket::ExternalConfigBuffer{.address = kX280ConfigAddr, .sender_is_l2cpu = true});
            dev_state.x280_socket->set_page_size(kX280PageSize);

            // Hart split (D8 Inc-1): nread reader harts drain L1 rings into per-(core,risc) LIM
            // mirrors; hart[nread] = collect (merges all mirrors into one SPSC); hart[nread+1] =
            // relay (single SPSC -> D2H FIFO). 2 read + 1 collect + 1 relay = 4 = all X280 harts.
            constexpr uint64_t kX280NRead = 2;   // reader harts (0..nread-1)
            constexpr uint64_t kX280NHarts = 4;  // nread readers + 1 collect + 1 relay
            std::vector<uint8_t> params(64, 0),
                results(384, 0);  // covers relay+reader+collect RES + probes 0x130/0x138
            auto pk = [&](size_t off, uint64_t val) { std::memcpy(params.data() + off, &val, 8); };
            pk(0x00, kX280ConfigAddr);
            pk(0x08, static_cast<uint64_t>(pc.x));
            pk(0x10, static_cast<uint64_t>(pc.y));
            pk(0x18, prof_l1);
            pk(0x20, num_cores);
            pk(0x28, 0);  // P_STOP = 0: run continuously until shutdown
            pk(0x30, kX280NRead);
            pk(0x38, kX280NHarts);
            zx.write_block(params.data(), (uint32_t)params.size(), kX280MboxParams);
            zx.write_block(results.data(), (uint32_t)results.size(), kX280MboxResults);
            // Pre-zero the mirror + single-SPSC control so every ring starts clean (no init race).
            // MIRRORCTL @ 0x08018000: 16 B/mirror (head,tail) for num_cores*5 mirrors — zero the whole
            // 12 KiB region (covers up to MAX_CORES=140). SINGLECTL @ 0x0801C000: prod/cons/collect_done/
            // reader_done. (Config md at 0x0801B000 sits between them and is written above, untouched.)
            std::vector<uint8_t> mirrorctl_zero(0x3000, 0);  // [0x08018000, 0x0801B000)
            zx.write_block(mirrorctl_zero.data(), (uint32_t)mirrorctl_zero.size(), 0x08018000ull);
            std::vector<uint8_t> singlectl_zero(256, 0);
            zx.write_block(singlectl_zero.data(), (uint32_t)singlectl_zero.size(), 0x0801C000ull);
            if (num_cores > 140) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] X280: num_cores={} exceeds MAX_CORES=140 -- mirrors would "
                    "overrun LIM; drainer will refuse to start (no DONE_MAGIC).",
                    num_cores);
            }
            dev_state.x280_params_addr = kX280MboxParams;

            zx.set_reset_vectors(profiler::X280_LIM_BASE);
            zx.set_pll(kX280PllMhz);
            zx.release_reset();

            // Fast-fail liveness check: poll profzone's main()-entry heartbeat (RES @ params+0x70)
            // for a few ms. If the core never writes it, the X280 isn't executing — on a fresh board
            // this is almost always because the L3 LIM ECC was never primed, so profzone's stores
            // fault silently.
            constexpr uint64_t kX280HbMainMagic = 0xB007ULL;
            uint64_t hb = 0;
            for (int i = 0; i < 300 && hb != kX280HbMainMagic; i++) {
                hb = zx.lim_rd_u64(dev_state.x280_params_addr + 0x70);
                if (hb != kX280HbMainMagic) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            // --- Auto-recover an unprimed X280 (ON by default) ---
            // The usual reason for a dead X280 on a fresh/cold-booted board is that its L3 LIM ECC was
            // never primed, so the drainer FW's stores fault. By default (opt out with
            // TT_METAL_X280_NO_AUTOPRIME) we prime the ECC and warm-reset the chip to activate it
            // (WayEnable is increase-only, so ONLY an ASIC/warm reset clears it). The warm reset
            // re-enumerates the PCIe device, so metal cannot continue in-process; we prime, reset, and
            // EXIT with an actionable message — the NEXT run boots cleanly. A /tmp marker guards
            // against a reset loop. TT_METAL_X280_FORCE_PRIME forces the path once even on a healthy
            // board, to exercise the prime+reset mechanics.
            {
                auto truthy = [](const char* s) {
                    return s && (s[0] == '1' || s[0] == 't' || s[0] == 'T' || s[0] == 'y' || s[0] == 'Y');
                };
                const bool force = truthy(std::getenv("TT_METAL_X280_FORCE_PRIME"));
                const bool autoprime = !truthy(std::getenv("TT_METAL_X280_NO_AUTOPRIME")) || force;
                const std::string marker = "/tmp/tt_x280_autoprime_dev" + std::to_string(device_id);
                const bool already_tried = std::ifstream(marker).good();
                if ((hb != kX280HbMainMagic || force) && autoprime && !already_tried) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: X280 unprimed (heartbeat=0x{:x}){}; auto-priming L3 "
                        "LIM ECC and warm-resetting the chip. This run will EXIT — rerun afterwards.",
                        device_id,
                        hb,
                        force ? " [forced]" : "");
                    bool primed_ok = false;
                    std::string fail_reason;
                    try {
                        {
                            std::ofstream(marker).put('1');
                        }  // loop guard: record that we tried this cycle
                        zx.assert_reset();
                        zx.prime_lim_ecc();
                        primed_ok = tt::umd::WarmReset::warm_reset_chip_id({static_cast<int>(device_id)});
                        if (!primed_ok) {
                            fail_reason = "warm_reset returned false";
                        }
                    } catch (const std::exception& e) {
                        fail_reason = e.what();
                    }
                    // The warm reset re-enumerates the chip out from under metal, so we hard-exit here —
                    // which skips buffered-log flushing. Print the outcome DIRECTLY to stderr so the
                    // RERUN instruction is guaranteed visible regardless of the logger's buffering.
                    std::fflush(stdout);
                    if (primed_ok) {
                        std::fprintf(
                            stderr,
                            "\n"
                            "================================================================================\n"
                            "  X280 (device %d): L3 LIM ECC was unprimed -> auto-primed + chip warm-reset.\n"
                            "  This is a one-time-per-cold-power-cycle step. The chip has been reset, so\n"
                            "  this run is stopping now.  >>> RERUN your program <<<  and the X280\n"
                            "  kernel-zone profiler will start normally.\n"
                            "================================================================================\n\n",
                            static_cast<int>(device_id));
                    } else {
                        std::fprintf(
                            stderr,
                            "\n"
                            "================================================================================\n"
                            "  X280 (device %d): auto-prime/warm-reset FAILED (%s).\n"
                            "  The chip may be in an undefined state -- run `tt-smi -r %d` manually, then\n"
                            "  rerun. (Set TT_METAL_X280_NO_AUTOPRIME=1 to disable this auto-recovery.)\n"
                            "================================================================================\n\n",
                            static_cast<int>(device_id),
                            fail_reason.c_str(),
                            static_cast<int>(device_id));
                    }
                    std::fflush(stderr);
                    std::_Exit(primed_ok ? 75 : 1);  // 75=EX_TEMPFAIL: "transient, rerun"; 1=hard failure
                }
            }
            // --- All-hart boot verification + retry ----------------------------------------------
            // hart 0's heartbeat only proves hart 0 started (and the ECC prime took). The X280's
            // 4-hart release_reset is intermittently flaky: a reader/collect/relay hart can fail to
            // start, which SILENTLY cripples the drainer -- its cores never drain, workers block on
            // full rings, and trace replay wedges (host hangs in finish). Verify EVERY hart reached
            // its work loop (per-hart stage 3 @ RESULTS+0x100+h*8) and, if any is missing, re-release
            // reset and retry; refuse to run a crippled drainer.
            if (hb == kX280HbMainMagic) {
                const uint64_t kHartStageBase = kX280MboxResults + 0x100;
                constexpr int kX280BootRetries = 3;
                auto all_harts_ready = [&]() {
                    for (uint32_t h = 0; h < kX280NHarts; h++) {
                        if (zx.lim_rd_u64(kHartStageBase + h * 8) < 3) {
                            return false;
                        }
                    }
                    return true;
                };
                auto wait_harts = [&]() {
                    for (int i = 0; i < 300 && !all_harts_ready(); i++) {
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                    return all_harts_ready();
                };
                auto stages_str = [&]() {
                    std::string s;
                    for (uint32_t h = 0; h < kX280NHarts; h++) {
                        if (h) {
                            s += ",";
                        }
                        s += std::to_string(zx.lim_rd_u64(kHartStageBase + h * 8));
                    }
                    return s;
                };
                for (int retry = 0; !wait_harts() && retry < kX280BootRetries; retry++) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: X280 not all harts started (stages [{}] want "
                        "all>=3) -- re-releasing reset (retry {}/{})",
                        device_id,
                        stages_str(),
                        retry + 1,
                        kX280BootRetries);
                    zx.assert_reset();
                    std::vector<uint8_t> mctl_zero(0x3000, 0), sctl_zero(256, 0), stg_zero(kX280NHarts * 8, 0);
                    zx.write_block(mctl_zero.data(), (uint32_t)mctl_zero.size(), 0x08018000ull);
                    zx.write_block(sctl_zero.data(), (uint32_t)sctl_zero.size(), 0x0801C000ull);
                    zx.write_block(stg_zero.data(), (uint32_t)stg_zero.size(), kHartStageBase);
                    zx.release_reset();
                    hb = 0;
                    for (int i = 0; i < 300 && hb != kX280HbMainMagic; i++) {
                        hb = zx.lim_rd_u64(dev_state.x280_params_addr + 0x70);
                        if (hb != kX280HbMainMagic) {
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        }
                    }
                }
                if (!all_harts_ready()) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: X280 harts still not all up after {} retries "
                        "(stages [{}]) -- refusing to run a crippled drainer (no X280 capture this run)",
                        device_id,
                        kX280BootRetries,
                        stages_str());
                    hb = 0;  // force the degraded path below
                } else {
                    log_info(
                        tt::LogMetal, "[Real-time profiler] Device {}: X280 all {} harts up", device_id, kX280NHarts);
                }
            }
            if (hb != kX280HbMainMagic) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: X280 drainer FW did not start (l2cpu {}, "
                    "heartbeat=0x{:x}). The L2CPU's L3 LIM ECC is likely not primed — but auto-prime was "
                    "either disabled (TT_METAL_X280_NO_AUTOPRIME) or already attempted this power cycle "
                    "without success (the X280 may be faulty; a manual `tt-smi -r` is advised). Continuing "
                    "without X280 kernel-zone capture (the DRAM-based device profiler covers kernel zones "
                    "as a fallback while TT_METAL_DEVICE_PROFILER is set).",
                    device_id,
                    kL2CpuIndex,
                    hb);
                zx.assert_reset();
                dev_state.x280_socket.reset();
                dev_state.x280_driver.reset();
                dev_state.x280_active = false;
                // No drainer => the cores' SPSC profiler rings will never be emptied. Tell every
                // worker to drop-not-block so a full ring can't deadlock the workload.
                try {
                    broadcast_profiler_terminate(x280_cluster, device_id, dev_state.device, prof_l1);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: failed to broadcast PROFILER_TERMINATE after X280 "
                        "boot failure: {}",
                        device_id,
                        e.what());
                }
            } else {
                dev_state.x280_active = true;
                // NB: do NOT set x280_dev_ = &dev_state here. `dev_state` is the caller's loop LOCAL,
                // which is immediately std::move'd into devices_ (and destroyed) after this returns --
                // the pointer would dangle at a moved-from stack object, whose moved-from unordered_map
                // reads as garbage/zero bucket_count in run_consumer (SIGFPE on `% 0`, or SIGSEGV once
                // the frame is reused). x280_dev_ is set post-loop, once devices_ is fully built + stable.
                // Clear the auto-prime loop guard: a clean boot means the prime (if any) took.
                std::remove(("/tmp/tt_x280_autoprime_dev" + std::to_string(device_id)).c_str());
                log_info(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: booted X280 kernel-zone drainer (l2cpu {}, "
                    "{} cores, prof_l1=0x{:x}, pcie=({},{}))",
                    device_id,
                    kL2CpuIndex,
                    num_cores,
                    prof_l1,
                    pc.x,
                    pc.y);
            }
        }
    } catch (const std::exception& e) {
        dev_state.x280_active = false;
        dev_state.x280_socket.reset();
        dev_state.x280_driver.reset();
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {}: X280 kernel-zone drainer boot failed ({}); continuing without it.",
            device_id,
            e.what());
        // No drainer => stop the cores blocking on full profiler rings (see boot-failure path above).
        try {
            const uint64_t prof_l1_t = MetalContext::instance(context_id_)
                                           .hal()
                                           .get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
            broadcast_profiler_terminate(
                MetalContext::instance(context_id_).get_cluster(), device_id, dev_state.device, prof_l1_t);
        } catch (const std::exception& e2) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {}: failed to broadcast PROFILER_TERMINATE after X280 boot "
                "exception: {}",
                device_id,
                e2.what());
        }
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
        // Pre-create the per-core Tracy contexts NOW (init, rings empty) from the boot-time coord
        // map, so the receiver thread never creates a context (~ms) lazily during draining — which
        // would block the socket drain and back-pressure the X280 into stalling the compute cores.
        if (dev_state.x280_active) {
            std::vector<std::pair<uint32_t, uint32_t>> worker_noc0;
            worker_noc0.reserve(dev_state.x280_virt_to_noc0.size());
            for (const auto& [virt, noc0] : dev_state.x280_virt_to_noc0) {
                worker_noc0.push_back(noc0);
            }
            tracy_handler_->PreCreateContexts(dev_state.chip_id, worker_noc0);
        }
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

// EXPERIMENT: raw-marker stall decoder. In DROP mode the FW bulk-flushes raw 2-word markers (no
// reshape, 8/64B-page). Rather than discard them, decode each marker and count PROFILER_STALL_ZONE
// (id 0x7FFF) START/END events -- the producer's self-measured back-pressure -- with no core info.
// This is the correct perturbation signal (device-side), which the reader "wall" does NOT capture.
namespace {
struct X280RawStall {
    uint64_t markers = 0, stall_start = 0, stall_end = 0, paired = 0, dur_sum = 0, dur_max = 0;
    uint64_t prev_start_ts = 0;
    bool have_prev = false;
    // STICKY_META tally + a sample of the first decoded one (validates the FW packet format).
    uint64_t sticky = 0;
    uint32_t s_cx = 0, s_cy = 0, s_risc = 0, s_id = 0;
    bool have_sticky_sample = false;
    uint32_t sticky_risc_mask = 0;  // bit r set if a sticky with risc==r was seen (expect 0x1F = all 5)
};
X280RawStall g_x280_raw;  // single receiver thread -> no lock
// PROTO diagnostic: total WorkerZoneWire records the receiver PUBLISHED to the BroadcastRing. Compared
// against the consumer's ring-dropped at shutdown to tell a real over-publish from a bogus drop count.
uint64_t g_x280_published = 0;  // single receiver thread -> no lock
// PROBE (b): total D2H pages the host READ + drain calls. If pages_read >> device-relayed pages, the
// host over-reads the FIFO (the ~496x re-read behind the 430M over-publish).
uint64_t g_x280_pages_read = 0;
uint64_t g_x280_drain_calls = 0;
}  // namespace

uint32_t RealtimeProfilerManager::drain_x280_device(DeviceState& dev_state) {
    if (!dev_state.x280_active || !dev_state.x280_socket) {
        return 0;
    }
    ZoneNamedN(z_x280_draincall, "X280-DrainCall", true);  // every active visit; SockRead fires only when avail>0
    namespace exp = tt::tt_metal::experimental;
    // Drain a big batch per call so the host frees FIFO room fast enough to keep up with the
    // multi-hart relay (256 was the limiter once the readers were parallelized -> the 1 MB FIFO
    // filled and the relay reserve-stalled). 4096 pages = 256 KB/read.
    static constexpr uint32_t kX280BatchPages = 4096;
    constexpr uint32_t page_words = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);

    const uint32_t avail = dev_state.x280_socket->pages_available();
    if (avail == 0) {
        return 0;
    }
    // BATCHED D2H drain: read up to kX280BatchPages in ONE read() and ack them ALL at once, so
    // FIFO room is freed in bulk and the X280 stops reserve-stalling. The per-page enrich+publish
    // below runs AFTER this bulk ack, so a slow Tracy push no longer gates the FIFO drain.
    // Reused across calls (single receiver thread) -- avoids a per-call heap allocation.
    static std::vector<uint32_t> x280_page_buf(static_cast<size_t>(kX280BatchPages) * page_words);
    const uint32_t n = std::min(avail, kX280BatchPages);
    {
        ZoneScopedN("X280-SockRead");  // D2H FIFO read (clflush + memcpy) + ack that frees FIFO for the X280
        dev_state.x280_socket->read(x280_page_buf.data(), n);
    }
    g_x280_pages_read += n;  // PROBE (b): total pages read vs. device-relayed pages
    g_x280_drain_calls++;

    // EXPERIMENT (TT_METAL_X280_DROP=1): the read() above already drained + ACKed the pages, freeing FIFO
    // room for the X280. Skip all host-side decode/enrich/emit and just discard them. Pairs with the FW
    // raw-flush experiment (reader bulk-copies raw markers, no reshape) to isolate the drain-side (X280
    // reader/relay) cost from host processing. Non-functional: no Tracy zones are produced in this mode.
    static const bool x280_drop = (std::getenv("TT_METAL_X280_DROP") != nullptr);
    if (x280_drop) {
        // Decode raw 2-word markers (8 per 64B page) and tally PROFILER_STALL_ZONE (id 0x7FFF) events.
        // No emit. START/END are adjacent within a ring's contiguous segment, so naive pairing gives a
        // rough duration; the START count is the robust metric. Timestamps are device cycles (~1.35 GHz).
        for (uint32_t pg = 0; pg < n; pg++) {
            const uint32_t* page = x280_page_buf.data() + static_cast<size_t>(pg) * page_words;
            for (uint32_t m = 0; m + 1 < page_words; m += 2) {
                const uint32_t w0 = page[m];
                const uint32_t w1 = page[m + 1];
                if (!(w0 & 0x80000000u)) {
                    continue;  // valid bit clear -> stale staging padding
                }
                g_x280_raw.markers++;
                const uint32_t type = (w0 >> 28) & 0x7;  // same bits for markers + sticky
                if (type == kernel_profiler::STICKY_META) {
                    g_x280_raw.sticky++;
                    const uint32_t rr = (w0 >> 14) & 0x3F;
                    if (rr < 32) {
                        g_x280_raw.sticky_risc_mask |= (1u << rr);
                    }
                    if (!g_x280_raw.have_sticky_sample) {
                        g_x280_raw.s_cx = (w0 >> 24) & 0xF;
                        g_x280_raw.s_cy = (w0 >> 20) & 0xF;
                        g_x280_raw.s_risc = (w0 >> 14) & 0x3F;
                        g_x280_raw.s_id = w1;
                        g_x280_raw.have_sticky_sample = true;
                    }
                    continue;
                }
                const uint32_t tid = (w0 >> 12) & 0x7FFFF;
                if ((tid & 0xFFFF) != 0x7FFF) {
                    continue;  // not a stall zone
                }
                const uint64_t ts = (static_cast<uint64_t>(w0 & 0xFFF) << 32) | w1;
                if (((tid >> 16) & 0x7) == kernel_profiler::ZONE_START) {
                    g_x280_raw.stall_start++;
                    g_x280_raw.prev_start_ts = ts;
                    g_x280_raw.have_prev = true;
                } else {  // ZONE_END
                    g_x280_raw.stall_end++;
                    if (g_x280_raw.have_prev) {
                        const uint64_t d = ts - g_x280_raw.prev_start_ts;
                        g_x280_raw.dur_sum += d;
                        if (d > g_x280_raw.dur_max) {
                            g_x280_raw.dur_max = d;
                        }
                        g_x280_raw.paired++;
                        g_x280_raw.have_prev = false;
                    }
                }
            }
        }
        return n;
    }

    // DECODE the X280 stream (Inc-2a) and PUBLISH to the ring. No enrich/Tracy here -- a consumer thread
    // does both off the receiver. The COLLECT hart already RESHAPED each raw marker into a WorkerZoneWire
    // with STRUCTURAL identity (core_x/y, risc taken from the mirror index it drained) and packed 2 per
    // 64B page. So there's no raw decode and no STICKY_META forward-fill here (that was the orphan source):
    // just lift out the valid slots. Validity: the FW sets header.reserved=0xA5A5 on real records; a padded
    // slot has reserved=0 -> skip. `wz` persists across drain calls (single receiver thread).
    static std::vector<exp::WorkerZoneWire> wz;
    wz.clear();
    ZoneScopedN("X280-DecodeWZW");  // WorkerZoneWire slots -> ring (reshape already done on the X280 collect hart)
    constexpr uint16_t kWzValid = 0xA5A5;  // collect-hart validity sentinel
    const uint32_t kWzPerPage = (page_words * sizeof(uint32_t)) / sizeof(exp::WorkerZoneWire);  // 64B/28B = 2
    const auto* page_bytes = reinterpret_cast<const uint8_t*>(x280_page_buf.data());
    for (uint32_t pg = 0; pg < n; pg++) {
        const uint8_t* page = page_bytes + static_cast<size_t>(pg) * page_words * sizeof(uint32_t);
        for (uint32_t s = 0; s < kWzPerPage; s++) {
            exp::WorkerZoneWire rec;
            std::memcpy(&rec, page + s * sizeof(exp::WorkerZoneWire), sizeof(exp::WorkerZoneWire));
            if (rec.header.reserved != kWzValid) {
                continue;  // padded/stale slot -> skip
            }
            wz.push_back(rec);
        }
    }
    if (!wz.empty()) {
        // Publish to the ring; a consumer thread enriches + pushes to Tracy (drain_all_devices wakes it).
        ring_->writer().publish_batch(std::span<const exp::WorkerZoneWire>(wz));
        g_x280_published += wz.size();
    }
    return n;
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
            {
                ZoneScopedN("ProgRecDrain");  // program-record path (yusuf) -- if this dominates, it starves X280
                num_pages += drain_device_pages(dev_state, scan_sync_marker, page_buf, record_buf);
            }
            // X280 kernel-zone drainer runs as a PARALLEL path: it pushes markers straight to
            // Tracy via the profiler-packet callbacks (not the broadcast ring), so its pages are
            // counted here only to keep the receiver's backoff/wake heuristics responsive.
            num_pages += drain_x280_device(dev_state);
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
    namespace exp = tt::tt_metal::experimental;
    // PROTOTYPING: the ring carries X280 WorkerZoneWire records (program records disabled). This consumer
    // reads them OFF the receiver thread, enriches (virt->noc0 + zone name), and pushes to Tracy via the
    // packet callback -- moving the Tracy push off the receiver (the confirmed back-pressure). consumer.callback
    // (the program-record callback) is unused in this mode.
    std::vector<exp::WorkerZoneWire> records(
        std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * std::max<size_t>(devices_.size(), 1)));
    std::unordered_map<uint16_t, tracy::MarkerDetails> zone_names;  // per-thread; loaded once
    try {
        zone_names = loadZoneSourceLocationsHashesReadOnly();
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[Real-time profiler] X280 zone-name resolution failed: {}", e.what());
    }

    auto push_batch = [&](std::span<const exp::WorkerZoneWire> batch) {
        DeviceState* dev = x280_dev_;
        if (dev == nullptr) {
            return;  // no X280 device booted
        }
        for (const auto& w : batch) {
            const uint32_t ptype = (w.timer_id >> 16) & 0x7;
            if (ptype != kernel_profiler::ZONE_START && ptype != kernel_profiler::ZONE_END) {
                continue;
            }
            uint32_t noc0_x = w.core_x, noc0_y = w.core_y;
            // Defensive: skip the lookups if a map is genuinely empty (no mapping => keep fallback values,
            // same outcome find() would give). A truly-empty x280_virt_to_noc0 would also be a real
            // enrichment bug, so it's logged once below. (This is NOT what caused the historical SIGFPE --
            // that was a dangling x280_dev_ pointing at a moved-from DeviceState; fixed at init time.)
            if (!dev->x280_virt_to_noc0.empty()) {
                if (auto it = dev->x280_virt_to_noc0.find((static_cast<uint64_t>(w.core_x) << 32) | w.core_y);
                    it != dev->x280_virt_to_noc0.end()) {
                    noc0_x = it->second.first;
                    noc0_y = it->second.second;
                }
            } else {
                [[maybe_unused]] static std::once_flag warned;
                std::call_once(warned, [] {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] X280 consumer: x280_virt_to_noc0 is EMPTY -- noc0 enrichment "
                        "disabled (zones will carry virtual coords). This also avoids a find()-on-empty SIGFPE.");
                });
            }
            const uint16_t hash = static_cast<uint16_t>(w.timer_id & 0xFFFF);
            std::string_view name;
            if (hash == 0x7FFF) {  // PROFILER_STALL_ZONE_ID
                name = "X280-STALL";
            } else if (!zone_names.empty()) {
                if (auto it = zone_names.find(hash); it != zone_names.end()) {
                    name = it->second.marker_name;
                }
            } else {
                [[maybe_unused]] static std::once_flag warned_zn;
                std::call_once(warned_zn, [] {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] X280 consumer: zone_names is EMPTY -- zone names unresolved "
                        "(zones will be unnamed). This also avoids a find()-on-empty SIGFPE.");
                });
            }
            exp::WorkerZonePacket pkt{
                .chip_id = dev->chip_id,
                .core_virtual_x = w.core_x,
                .core_virtual_y = w.core_y,
                .core_noc0_x = noc0_x,
                .core_noc0_y = noc0_y,
                .risc = w.risc,
                .timer_id = hash,
                .name = name,
                .timestamp = (static_cast<uint64_t>(w.time_hi) << 32) | w.time_lo,
                .is_start = (ptype == kernel_profiler::ZONE_START),
            };
            exp::InvokeProfilerPacketCallbacks(exp::ProfilerPacketType::WorkerZone, &pkt);
        }
    };

    while (true) {
        // stop_consumer sets the stop mode then wakes, so waiting on a token sampled only inside wait() could miss that
        // wake and hang
        const auto token = consumer.reader.wait_token();
        const auto batch = consumer.reader.read_batch(records);
        const ConsumerStopMode stop_mode = consumer.stop_mode.load(std::memory_order_acquire);
        if (stop_mode == ConsumerStopMode::StopWithoutDrain) {
            break;
        }
        if (!batch.empty()) {
            push_batch(batch);
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

    // The SPSC kernel-profiler backend has worker+dispatch cores BLOCK in ring_ensure_room when their
    // zone ring is full. Once the X280 drainer stops advancing ring heads at teardown, a still-producing
    // core (typically a dispatch core) would block forever and wait_for_dispatch_cores would never see it
    // "done" -> the process hangs at close. Broadcast PROFILER_TERMINATE to the FULL logical grid first so
    // ring_ensure_room drops-not-blocks (lossless during the run; only teardown-phase markers are dropped),
    // BEFORE we P_STOP the drainer.
    if (!devices_.empty()) {
        const auto& hal = MetalContext::instance(context_id_).hal();
        auto& cluster = MetalContext::instance(context_id_).get_cluster();
        const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
        for (auto& dev_state : devices_) {
            if (dev_state.x280_active && dev_state.device) {
                broadcast_profiler_terminate(cluster, dev_state.chip_id, dev_state.device, prof_l1);
            }
        }
    }

    // Re-write ring_buffer->terminate as a safety net (dispatch_s already set it via the
    // profiler core's TERMINATE), then give the push kernel time to deliver the last PCIe page.
    for (auto& dev_state : devices_) {
        // Tell the X280 drainer to finish its current drain pass and exit its loop, so the
        // receiver's shutdown drain catches the last device-zone pages.
        if (dev_state.x280_active && dev_state.x280_driver) {
            try {
                dev_state.x280_driver->lim_wr_u64(dev_state.x280_params_addr + 0x28, 1);  // P_STOP
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to stop X280 on device {}: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
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
    // Drain-to-quiescence for the X280 path: the readers finish their L1 rings, the relay flushes all
    // LIM-staged pages into the D2H FIFO, and the (still-running) receiver thread drains them -- so no
    // staged marker is dropped at teardown. The relay writes DONE_MAGIC to its results slot once its
    // staging + the FIFO are empty; poll for it (bounded) before stopping the receiver / resetting.
    for (auto& dev_state : devices_) {
        if (!dev_state.x280_active || !dev_state.x280_driver) {
            continue;
        }
        constexpr uint64_t kX280DoneMagic = 0x20E50FFEE1ull;
        const uint64_t done_addr = dev_state.x280_params_addr + 0x40 + 0x18;  // RESULTS base + RES_DONE
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < deadline) {
            try {
                if (dev_state.x280_driver->lim_rd_u64(done_addr) == kX280DoneMagic) {
                    break;
                }
            } catch (const std::exception&) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
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

    // Park each X280 in reset now that its socket has been drained by the receiver's shutdown pass.
    for (auto& dev_state : devices_) {
        if (dev_state.x280_active && dev_state.x280_driver) {
            try {
                // Telemetry: profzone's result mailbox (results base = params_addr + 0x40).
                // relayed_pages@+0x00 = 64B D2H PAGES relayed (profzone increments `total` once per
                // page_copy, NOT per marker); each page carries up to 8 raw 2-word markers, so
                // marker-slots = pages*8 (>= real markers; last page of each ring segment is padded).
                // loops@+0x08 = drain-loop passes; stalls@+0x20 = reserve-spin iterations where the X280
                // was BLOCKED waiting for the host to free D2H FIFO room. High stalls => host drain is the
                // bottleneck; ~0 => the X280's own drain rate (or marker supply) is the limit.
                uint64_t relayed_pages = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x40);
                uint64_t loops = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x48);
                uint64_t reserve_stalls = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x60);
                log_info(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: X280 drainer relayed {} pages ({} B, up to {} marker-slots) "
                    "({} drain passes, {} reserve-stall spins waiting on host D2H FIFO)",
                    dev_state.chip_id,
                    relayed_pages,
                    relayed_pages * 64,
                    relayed_pages * 8,
                    loops,
                    reserve_stalls);
                // Relay time-split (X280 ~1 GHz => cycles ~= ns). empty-spin = wall - reserve - copy is
                // time the relay found the single SPSC empty (collect slower than D2H); reserve = time
                // blocked on the host D2H FIFO (host too slow); copy = page_copy + bytes_sent.
                {
                    uint64_t rwall = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xF0);
                    uint64_t rreserve = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xF8);
                    uint64_t rcopy = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x100);
                    uint64_t rempty = rwall - (rreserve + rcopy);
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: relay time-split (of {} ms wall): reserve-stall={} ms "
                        "(host FIFO), copy={} ms, empty-spin={} ms (single SPSC empty / collect slow)",
                        dev_state.chip_id,
                        rwall / 1000000,
                        rreserve / 1000000,
                        rcopy / 1000000,
                        rempty / 1000000);
                }
                // Collect-hart baseline (D8 Inc-1): how busy is the merge hart doing PURE DRAIN (no
                // reshape)? empty-spin = wall - copy = time round-robining idle mirrors; copy = time in
                // the mirror->single copy path. moved = markers merged into the single SPSC.
                {
                    uint64_t cmoved = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x108);
                    uint64_t cloops = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x110);
                    uint64_t cwall = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x118);
                    uint64_t cempty = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x120);
                    uint64_t ccopy = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x128);
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: collect-hart split (of {} ms wall): copy={} ms, "
                        "empty-spin={} ms (mirrors idle), moved {} markers, {} loops",
                        dev_state.chip_id,
                        cwall / 1000000,
                        ccopy / 1000000,
                        cempty / 1000000,
                        cmoved,
                        cloops);
                    // PROBE-a: disambiguate the collect-telemetry-0 puzzle.
                    //  collect_direct_loops (RES 0x130): collect wrote `loops` DIRECTLY to a fresh RES line.
                    //  relay_read_sentinel (RES 0x138): relay copied COLLECT_STATS entry sentinel (0xC0FFEE01).
                    // both nonzero+correct => collect ran & visible => cmoved should be right;
                    // direct nonzero but sentinel 0 => relay can't read COLLECT_STATS;
                    // direct 0 => collect telemetry code not executing.
                    uint64_t collect_direct_loops =
                        dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x170);
                    uint64_t relay_read_sentinel =
                        dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x178);
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: PROBE-a collect_direct_loops={} relay_read_sentinel=0x{:x} "
                        "(expect sentinel 0xc0ffee01)",
                        dev_state.chip_id,
                        collect_direct_loops,
                        relay_read_sentinel);
                }
                // Raw-marker stall decode (DROP mode only): device-side back-pressure straight from the
                // marker stream. ~1.35 GHz => cycles/1350 = us. START count is the robust metric.
                if (std::getenv("TT_METAL_X280_DROP") != nullptr) {
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: RAW stall decode: {} markers, stall START={}, END={}, "
                        "paired={}, dur avg={} us, max={} us",
                        dev_state.chip_id,
                        g_x280_raw.markers,
                        g_x280_raw.stall_start,
                        g_x280_raw.stall_end,
                        g_x280_raw.paired,
                        g_x280_raw.paired ? (g_x280_raw.dur_sum / g_x280_raw.paired) / 1350 : 0,
                        g_x280_raw.dur_max / 1350);
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: RAW sticky decode: {} STICKY_META packets; "
                        "risc-mask=0x{:x} (0x1F=all 5); sample core=({},{}) risc={} host_id={}",
                        dev_state.chip_id,
                        g_x280_raw.sticky,
                        g_x280_raw.sticky_risc_mask,
                        g_x280_raw.s_cx,
                        g_x280_raw.s_cy,
                        g_x280_raw.s_risc,
                        g_x280_raw.s_id);
                }
                // Reader ILP diagnostic: avg bulk-read segment width (words) per reader. Small (<~8)
                // => the round-robin drains rings to near-empty each visit, so the vector burst is
                // too narrow to pipeline (ILP inactive); large => wide bursts, ILP is doing work.
                for (int rh = 0; rh < 2; ++rh) {
                    uint64_t bulk_words = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xA0 + rh * 8);
                    uint64_t segs = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xB0 + rh * 8);
                    // Count-based (per-iteration rdcycle traps on the X280 => contaminates timing). One
                    // rdcycle pair gives an uncontaminated wall; counts give per-op cost. X280 ~1 GHz.
                    uint64_t wall = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xC0 + rh * 8);
                    uint64_t passes = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xD0 + rh * 8);
                    uint64_t polls = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0xE0 + rh * 8);
                    // Reader lap-guard drop counter (profzone RES(0x50+h*8) == params+0x90+h*8). The reader
                    // clamps head to tail-RING_CAP when the producer got >512 words ahead, DROPPING the
                    // oldest words -- and the oldest word in a dispatch is the FW ZONE_START. If this is
                    // NONZERO the "producer always blocks so tail-head<=512" invariant is being violated
                    // under back-pressure, and the reader-side drop is the root cause of the orphan ENDs.
                    uint64_t lap_dropped =
                        dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x90 + rh * 8);
                    uint64_t markers = bulk_words / 2;
                    double wall_ms = wall / 1e6;
                    log_info(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: reader {}: wall={} ms, {} markers => {} k/s; {} passes, "
                        "{} core-polls ({} ns/poll); {} segments (avg {} words/seg); LAP-DROPPED {} words "
                        "(want 0 -- nonzero => reader dropped oldest markers/STARTs under back-pressure)",
                        dev_state.chip_id,
                        rh,
                        (uint64_t)wall_ms,
                        markers,
                        wall_ms > 0 ? (uint64_t)(markers / wall_ms) : 0,
                        passes,
                        polls,
                        polls ? wall / polls : 0,
                        segs,
                        segs ? bulk_words / segs : 0,
                        lap_dropped);
                }
                dev_state.x280_driver->assert_reset();
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to reset X280 on device {}: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
        dev_state.x280_socket.reset();
        dev_state.x280_driver.reset();
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
            // PROTO diagnostic: records the lossy BroadcastRing dropped before this consumer read them.
            // Should be 0 now that the ring is sized for the device-marker backlog.
            log_info(
                tt::LogMetal,
                "[Real-time profiler] X280 consumer {}: ring-dropped {} records (receiver published {} total; "
                "PROBE-b: host read {} pages over {} drain calls -- if >> device-relayed pages, host over-reads)",
                handle,
                consumer->dropped,
                g_x280_published,
                g_x280_pages_read,
                g_x280_drain_calls);
        }
        consumers_.clear();
    }

    tracy_handler_.reset();
    // Clear activation state before destroying per-device records so concurrent
    // tt::IsProgramRealtimeProfilerActive() queries don't observe a chip mid-shutdown.
    for (const auto& dev_state : devices_) {
        tt::NotifyProgramRealtimeProfilerDeactivated(dev_state.chip_id);
        tt::NotifyProgramX280ProfilerDeactivated(dev_state.chip_id);  // no-op if X280 never won here
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
