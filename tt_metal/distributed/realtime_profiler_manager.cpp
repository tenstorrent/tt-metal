// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/realtime_profiler_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
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
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_metal.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/driver_atomics.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <common/TracySystem.hpp>
#include <llrt/tt_cluster.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/hal.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_host_clock.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal::distributed {

namespace {

// Minimum wall time between full init calibrations (run_sync + constructor SYNC_CHECK) and reuse of a cached
// calibration across a MeshDevice reopen, per physical chip.
constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

// Cadence of the free-running host<->device offset re-anchor servo. The init fit is taken at idle; under sustained
// load the device clock shifts ~9ppm (thermal/DVFS), so the fixed slope goes stale and the mapped host time drifts
// by staleness*(time since last re-anchor). Re-anchoring this often bounds that to ~2µs, vs the ~100µs a 60s
// interval accrued. Cheap: the one-shot device sync keeps each re-anchor drop-free even under peak record load.
constexpr auto kServoInterval = std::chrono::milliseconds(50);
// Hung-device backstop for the round-trip probe's busy-poll (measure_sync_rtt_ticks). A wall clock (relative to
// host_before) so it does not vary with a host's PCIe read latency, but the probe skips the clock read for the first
// kRttProbeHealthyPolls reads — far above any healthy handshake (a few to tens of reads) — so a real measurement never
// reads the clock inside the round trip it is timing; only a stalled device (which yields no measurement) reaches the
// checked region.
constexpr double kRttProbeTimeoutNs = 300'000.0;
constexpr uint32_t kRttProbeHealthyPolls = 128;

constexpr auto kFinishSyncResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kSyncResponsePollBackoff = std::chrono::microseconds(100);

// Last full init sync per chip, process-wide, to avoid repeating ~0.5s run_sync on every mesh open.
// Per-physical-chip calibration, cached across MeshDevice open/close so a rapid reopen can reuse the recent fit instead
// of re-running the expensive host-device sync. The device WALL_CLOCK is free-running (not reset on close), so the
// fitted frequency and device_cycle_offset stay valid; updated_at gates reuse to a recent-enough sync.
struct CachedCalibration {
    std::chrono::steady_clock::time_point updated_at;
    double sync_frequency = 0.0;
    double device_cycles_per_host_tick = 0.0;
    int64_t device_cycle_offset = 0;
    int64_t sync_rtt_ticks = 0;
};
std::mutex g_rt_profiler_init_sync_mu;
std::unordered_map<uint32_t, CachedCalibration> g_rt_profiler_calibration_by_chip;

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

// Result of evaluating whether the real-time profiler can be brought up on a device.
struct RealtimeProfilerEligibility {
    bool enabled = false;
    CoreCoord core;  // Only meaningful when enabled == true.
};

// Consolidated eligibility check; logs the disable reason. Evaluates against the device's owning context_id (not bare
// instance()) so a mock device isn't falsely enabled via the silicon DEFAULT_CONTEXT_ID fallback (#38445/#39849).
// Checks: not mock/emulated, MMIO-capable, IOMMU if 64-bit PCIe, fabric tensix datamover off, a tensix reserved and
// in-grid, kernels not nullified, L1 bank fits the layout.
RealtimeProfilerEligibility evaluate_realtime_profiler_eligibility(IDevice* device, ContextId context_id) {
    auto device_id = device->id();
    auto& metal = MetalContext::instance(context_id);
    const auto& hal = metal.hal();
    const auto& cluster = metal.get_cluster();
    auto& dispatch_core_manager = metal.get_dispatch_core_manager();

    // Gate mock/emulated targets: D2HSocket::init_host_buffer_hugepage dereferences a real PCIe hugepage absent there.
    if (cluster.is_mock_or_emulated()) {
        log_debug(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: target is mock or emulated; D2H sockets "
            "require a real PCIe hugepage that is not present in mock/emulated flows.",
            device_id);
        return {};
    }

    // Skip Simulator: ttsim kernels are too slow to meet run_sync's 2s poll deadline, burning ~30s/chip and deadlocking
    // finish_sync waiters on WH.
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

size_t RealtimeProfilerManager::publish_pages(
    const DeviceState& dev_state,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<tt::ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    auto is_record = [](const uint32_t* page) { return page[2] != 0 && page[3] != REALTIME_PROFILER_SYNC_MARKER_ID; };
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double sync_frequency = dev_state.sync_frequency;
    const int64_t half_rtt_ns =
        std::llround(static_cast<double>(dev_state.sync_rtt_ticks) * 0.5 * realtime_profiler_host_ns_per_tick());
    const uint64_t sync_error_ns =
        static_cast<uint64_t>(half_rtt_ns) + static_cast<uint64_t>(std::abs(dev_state.sync_tracking_error_ns));
    const tt::tt_metal::experimental::ProgramRealtimeClockSync clock_sync{dev_state.device_cycle_offset, sync_error_ns};
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
            clock_sync,
            data_collector->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])));
    }
    if (records.empty()) {
        return 0;
    }
    num_published_records_.fetch_add(records.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_->writer().publish_batch(std::span<const tt::ProgramRealtimeRecord>(records));
    return records.size();
}

bool RealtimeProfilerManager::has_active_finish_sync() const {
    for (const auto& dev_state : devices_) {
        if (dev_state.finish_sync_phase != DeviceState::FinishSyncPhase::Idle) {
            return true;
        }
    }
    return false;
}

void RealtimeProfilerManager::write_sync_timestamp(RealtimeProfilerManager::DeviceState& dev_state, uint32_t value) {
    if (dev_state.sync_tlb != nullptr) {
        dev_state.sync_tlb->write32(dev_state.sync_host_ts_addr, value);
        tt_driver_atomics::sfence();
    } else {
        std::vector<uint32_t> data = {value};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device, dev_state.realtime_profiler_core, dev_state.sync_host_ts_addr, data, CoreType::WORKER);
    }
}

void RealtimeProfilerManager::start_finish_syncs(std::chrono::steady_clock::time_point now) {
    bool started = false;
    for (auto& dev_state : devices_) {
        if (dev_state.finish_sync_phase != DeviceState::FinishSyncPhase::Idle) {
            continue;
        }
        const bool interval_elapsed =
            !dev_state.last_finish_sync_at.has_value() || now - *dev_state.last_finish_sync_at >= kServoInterval;
        if (!interval_elapsed) {
            continue;
        }
        try {
            dev_state.sync_host_time_before = realtime_profiler_host_timestamp();
            if (++dev_state.sync_seq == 0) {
                dev_state.sync_seq = 1;
            }
            const uint32_t host_time_id = dev_state.sync_seq;
            write_sync_timestamp(dev_state, host_time_id);
            if (dev_state.ack_host_ptr != nullptr) {
                const int64_t rtt_ticks =
                    measure_sync_rtt_ticks(dev_state, dev_state.sync_host_time_before, host_time_id);
                if (rtt_ticks >= 0) {
                    // Token observed. device_time was published to L1 before the token, so read it directly and
                    // re-anchor synchronously — a re-anchor lands every servo tick regardless of FIFO/marker latency,
                    // so no AwaitingResponse and no dependence on the marker path that skips ticks under record-push
                    // load.
                    std::vector<uint32_t> dt(2, 0);
                    tt::tt_metal::detail::ReadFromDeviceL1(
                        dev_state.device,
                        dev_state.realtime_profiler_core,
                        dev_state.sync_device_time_addr,
                        2 * sizeof(uint32_t),
                        dt,
                        CoreType::WORKER);
                    const uint64_t device_time = (static_cast<uint64_t>(dt[1]) << 32) | static_cast<uint64_t>(dt[0]);
                    dev_state.sync_rtt_ticks = rtt_ticks;
                    // Midpoint of the round-trip bracket: minimax placement, error <= RTT/2 without assuming symmetry.
                    reanchor_device_cycle_offset(
                        dev_state, dev_state.sync_host_time_before + rtt_ticks / 2, device_time);
                    cache_calibration(dev_state);
                    dev_state.last_finish_sync_at = now;
                } else {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {} sync round-trip probe timed out; keeping previous mapping",
                        dev_state.chip_id);
                }
            } else {
                // Fallback (no host ACK pin): the FIFO sync marker drives the re-anchor asynchronously in the drain.
                dev_state.finish_sync_deadline = now + kFinishSyncResponseTimeout;
                dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::AwaitingResponse;
                started = true;
            }
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Failed to start sync for device {}: {}",
                dev_state.chip_id,
                e.what());
            continue;
        }
    }
    finish_sync_busy_.store(started || has_active_finish_sync(), std::memory_order_release);
}

void RealtimeProfilerManager::advance_finish_sync(DeviceState& dev_state, std::chrono::steady_clock::time_point now) {
    if (dev_state.finish_sync_phase == DeviceState::FinishSyncPhase::AwaitingResponse &&
        now > dev_state.finish_sync_deadline) {
        log_warning(tt::LogMetal, "[Real-time profiler] Sync check timed out for device {}", dev_state.chip_id);
        dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::Idle;
        finish_sync_busy_.store(has_active_finish_sync(), std::memory_order_release);
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

    auto& metal_context = MetalContext::instance(context_id_);
    data_collector_ = metal_context.data_collector().get();
    realtime_profiler_service_ = metal_context.realtime_profiler_service().get();
    TT_FATAL(realtime_profiler_service_ != nullptr, "Missing context-wide real-time profiler service");

    run_init_sync();

    realtime_profiler_service_->attach_ring(*ring_, max_consumer_batch_records);
    ring_attached_ = true;

    // Background receiver thread that polls all device sockets round-robin
    try {
        receiver_thread_ = std::thread(&RealtimeProfilerManager::run_receiver, this);
    } catch (...) {
        realtime_profiler_service_->detach_ring(*ring_);
        ring_attached_ = false;
        throw;
    }
}

void RealtimeProfilerManager::initialize_devices(const std::shared_ptr<MeshDevice>& mesh_device) {
    // HAL offsets are the same for all devices (same arch).
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
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
    uint32_t sync_host_timestamp_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_host_timestamp);
    uint32_t sync_ack_enc_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_ack_pcie_xy_enc);
    uint32_t sync_ack_lo_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_ack_host_addr_lo);
    uint32_t sync_ack_hi_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_ack_host_addr_hi);
    uint32_t sync_ack_device_time_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::sync_ack_device_time);
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

        dev_state.sync_host_ts_addr = realtime_profiler_base_addr + sync_host_timestamp_offset;
        dev_state.sync_device_time_addr = realtime_profiler_base_addr + sync_ack_device_time_offset;

        // Blackhole maps the full device address space through static TLBs, so the profiler core's window is
        // resolved once and stays valid for the manager's lifetime; hold it to write the sync timestamp with a
        // single MMIO store. Other archs use dynamic TLBs (reconfigured per access), so sync_tlb stays null and
        // write_sync_timestamp falls back to WriteToDeviceL1. get_tlb_window only looks up the pre-mapped window
        // (throws if absent), so guard it.
        if (hal.get_arch() == tt::ARCH::BLACKHOLE) {
            try {
                CoreCoord rt_virtual = device->virtual_core_from_logical_core(realtime_profiler_core, CoreType::WORKER);
                auto* tlb_manager = MetalContext::instance(context_id_)
                                        .get_cluster()
                                        .get_driver()
                                        ->get_chip(device_id)
                                        ->get_tlb_manager();
                if (tlb_manager != nullptr) {
                    dev_state.sync_tlb = tlb_manager->get_tlb_window(tt_xy_pair(rt_virtual.x, rt_virtual.y));
                }
            } catch (const std::exception& e) {
                log_debug(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: no TLB window for the profiler core ({}); sync uses "
                    "WriteToDeviceL1",
                    device_id,
                    e.what());
            }
        }

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

        // Ring buffer (BRISC->NCRISC handoff) at a fixed carve-out offset; not Buffer::create'd since the core is off
        // the L1 bank table.
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

        // Zero realtime_profiler_msg_t before launch: stale L1 values misbehave at BRISC/NCRISC boot (garbage socket
        // config, premature sync, phantom marker, corrupt state machine).
        {
            const uint32_t profiler_msg_size = factory.size_of<realtime_profiler_msgs::realtime_profiler_msg_t>();
            const uint32_t profiler_msg_words = profiler_msg_size / sizeof(uint32_t);
            std::vector<uint32_t> zero_msg(profiler_msg_words, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, realtime_profiler_base_addr, zero_msg, CoreType::WORKER);
        }

        // Allocate a device-reachable sysmem word the NCRISC NOC-writes the sync token into (device->host, bypassing
        // the record FIFO), so the host times the handshake round trip by polling its own memory. It comes from the
        // same CQ sysmem region the D2H socket draws from, so the profiler core reaches it over the PCIe NOC-1 path it
        // already uses for records; a separately host-pinned buffer is only reachable over NOC 0, so the NCRISC's
        // writes never land. If the region is exhausted, ack_host_ptr stays null and sync falls back to the FIFO
        // marker.
        {
            auto [ack_host_ptr, ack_dev_addr] = device->sysmem_manager().allocate_region(sizeof(uint32_t));
            if (ack_host_ptr != nullptr) {
                dev_state.ack_host_ptr = static_cast<volatile uint32_t*>(ack_host_ptr);
                *const_cast<uint32_t*>(dev_state.ack_host_ptr) = 0;
                // Non-zero sentinel telling the device an ACK word exists; the NCRISC writes with its own NOC-1 PCIe
                // encoding, so this value is not used as an address component.
                std::vector<uint32_t> enc = {1};
                std::vector<uint32_t> lo = {ack_dev_addr};
                std::vector<uint32_t> hi = {0};
                tt::tt_metal::detail::WriteToDeviceL1(
                    device,
                    realtime_profiler_core,
                    realtime_profiler_base_addr + sync_ack_enc_offset,
                    enc,
                    CoreType::WORKER);
                tt::tt_metal::detail::WriteToDeviceL1(
                    device,
                    realtime_profiler_core,
                    realtime_profiler_base_addr + sync_ack_lo_offset,
                    lo,
                    CoreType::WORKER);
                tt::tt_metal::detail::WriteToDeviceL1(
                    device,
                    realtime_profiler_core,
                    realtime_profiler_base_addr + sync_ack_hi_offset,
                    hi,
                    CoreType::WORKER);
            }
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
    const auto init_throttle_now = std::chrono::steady_clock::now();
    std::vector<bool> skip_init_sync_check(devices_.size(), false);
    std::vector<size_t> init_run_sync_indices;
    init_run_sync_indices.reserve(devices_.size());

    // Run our own host-device sync (device profiler's SyncInfo masks the high word to 12 bits, shifting RT zones by
    // hours); skip recently init-synced chips.
    for (size_t di = 0; di < devices_.size(); ++di) {
        auto& dev_state = devices_[di];
        std::optional<CachedCalibration> cached;
        {
            std::lock_guard<std::mutex> lock(g_rt_profiler_init_sync_mu);
            const auto it = g_rt_profiler_calibration_by_chip.find(dev_state.chip_id);
            if (it != g_rt_profiler_calibration_by_chip.end() &&
                init_throttle_now - it->second.updated_at < kRtProfilerMinSyncInterval) {
                cached = it->second;
            }
        }

        if (cached.has_value()) {
            // Reuse the recent fit instead of re-syncing: WALL_CLOCK is free-running across reopen, so both the fitted
            // frequency and device_cycle_offset stay valid. Skips run_sync and the constructor SYNC_CHECK entirely
            // (no device traffic); periodic finish-path syncs still re-anchor and refresh the cache during the session.
            dev_state.sync_frequency = cached->sync_frequency;
            dev_state.device_cycles_per_host_tick = cached->device_cycles_per_host_tick;
            dev_state.device_cycle_offset = cached->device_cycle_offset;
            dev_state.sync_rtt_ticks = cached->sync_rtt_ticks;
            dev_state.first_timestamp = 0;
            dev_state.sync_host_start = realtime_profiler_host_timestamp();
            dev_state.last_finish_sync_at = init_throttle_now;
            skip_init_sync_check[di] = true;
            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: reusing cached calibration (last sync within {}s), skipping init "
                "run_sync and constructor SYNC_CHECK",
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
    });

    std::vector<size_t> init_sync_check_indices;
    init_sync_check_indices.reserve(devices_.size());
    for (size_t di = 0; di < devices_.size(); ++di) {
        if (!skip_init_sync_check[di]) {
            init_sync_check_indices.push_back(di);
        }
    }
    parallel_for_each_device_index(init_sync_check_indices, [&](size_t di) {
        auto& dev_state = devices_[di];

        std::this_thread::sleep_for(kConstructorSyncCheckDelay);

        const int64_t sync_check_host_anchor = realtime_profiler_host_timestamp();
        if (++dev_state.sync_seq == 0) {
            dev_state.sync_seq = 1;
        }
        const uint32_t host_time_id = dev_state.sync_seq;
        write_sync_timestamp(dev_state, host_time_id);
        const int64_t rtt_ticks = dev_state.ack_host_ptr != nullptr
                                      ? measure_sync_rtt_ticks(dev_state, sync_check_host_anchor, host_time_id)
                                      : -1;

        auto sc_deadline = std::chrono::steady_clock::now() + kConstructorSyncCheckTimeout;
        bool sc_got_response = false;
        while (std::chrono::steady_clock::now() < sc_deadline) {
            if (dev_state.socket->pages_available() > 0) {
                sc_got_response = true;
                break;
            }
            std::this_thread::sleep_for(kSyncResponsePollBackoff);
        }

        if (sc_got_response) {
            std::vector<uint32_t> sync_page(RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t));
            dev_state.socket->read(sync_page.data(), 1);
            uint64_t device_time = (static_cast<uint64_t>(sync_page[0]) << 32) | sync_page[1];

            // First offset real records use (run_sync fits only the slope); midpoint anchor, see the servo re-anchor.
            const int64_t anchor_rtt_ticks = rtt_ticks >= 0 ? rtt_ticks : 0;
            reanchor_device_cycle_offset(dev_state, sync_check_host_anchor + anchor_rtt_ticks / 2, device_time);
            if (rtt_ticks >= 0) {
                dev_state.sync_rtt_ticks = rtt_ticks;
            }
            cache_calibration(dev_state);

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

RealtimeProfilerManager::DrainCounts RealtimeProfilerManager::drain_device_pages(
    DeviceState& dev_state,
    bool scan_sync_marker,
    std::vector<uint32_t>& page_buf,
    std::vector<tt::ProgramRealtimeRecord>& record_buf) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    uint32_t available = dev_state.socket->pages_available();
    if (available > peak_fifo_pages_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_.store(available, std::memory_order_relaxed);
    }
    fifo_pages_window_max_ = std::max(fifo_pages_window_max_, available);
    if (available >= RealtimeProfilerRuntimeSizes::fifo_pages && !dev_state.fifo_reached_capacity) {
        dev_state.fifo_reached_capacity = true;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} D2H FIFO reached capacity ({} pages); profiler data may be dropped",
            dev_state.chip_id,
            available);
    }
    if (available == 0) {
        return {};
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
            // Anchor device_time at the host instant we observed the round-trip ACK (sync_host_time_before + RTT), not
            // at sync_host_time_before. The reserved core captures device_time partway through the round trip, after a
            // variable pre-service loop delay; that forward latency would otherwise be baked into the offset and make
            // it lurch whenever a handshake is slow. RTT = forward + backward, and the backward leg (ACK write + host
            // poll) is unaffected by that delay, so adding the full RTT cancels the variable forward latency, leaving
            // only a ~constant backward-leg bias (which cancels in tracking).
            const int64_t host_anchor = dev_state.sync_host_time_before + dev_state.sync_rtt_ticks;
            reanchor_device_cycle_offset(dev_state, host_anchor, device_time);
            cache_calibration(dev_state);
            dev_state.last_finish_sync_at = std::chrono::steady_clock::now();
            dev_state.finish_sync_phase = DeviceState::FinishSyncPhase::Idle;
            finish_sync_busy_.store(has_active_finish_sync(), std::memory_order_release);
            break;
        }
    }
    const size_t records = publish_pages(dev_state, page_buf.data(), num_pages_to_read, record_buf);
    return {num_pages_to_read, records};
}

uint64_t RealtimeProfilerManager::run_receiver_loop() {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * kPageWords);
    std::vector<tt::ProgramRealtimeRecord> record_buf;
    record_buf.reserve(kMaxSocketPagesPerRead);
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;
    auto last_servo_tick = std::chrono::steady_clock::now();
#if defined(TRACY_ENABLE) && TT_TRACY_CATEGORY_RT_PROFILER
    uint64_t last_diagnostics_records = 0;
    // Emit the RT-profiler plots (no-op without a live Tracy server). Primed here so the init-sync values show from the
    // first frame; then refreshed on each servo tick.
    auto emit_diagnostics_plots = [&]() {
        if (!TTTracyConnected()) {
            return;
        }
        constexpr double kServoSecs = std::chrono::duration<double>(kServoInterval).count();
        const uint64_t records = num_published_records_.load(std::memory_order_relaxed);
        TTPlotD(
            RT_PROFILER,
            "RT profiler publish rate (rec/s)",
            static_cast<double>(records - last_diagnostics_records) / kServoSecs);
        TTPlotD(RT_PROFILER, "RT profiler D2H FIFO pages", static_cast<double>(fifo_pages_window_max_));
        fifo_pages_window_max_ = 0;
        int64_t worst_sync_error_ns = 0;
        for (const auto& dev_state : devices_) {
            const int64_t half_rtt_ns = std::llround(
                static_cast<double>(dev_state.sync_rtt_ticks) * 0.5 * realtime_profiler_host_ns_per_tick());
            worst_sync_error_ns =
                std::max(worst_sync_error_ns, half_rtt_ns + std::abs(dev_state.sync_tracking_error_ns));
        }
        PlotD(RT_PROFILER, "RT sync error (us)", static_cast<double>(worst_sync_error_ns) / 1000.0);
        last_diagnostics_records = records;
    };
    emit_diagnostics_plots();
#endif
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const bool servo_ticked = now - last_servo_tick >= kServoInterval;
        if (servo_ticked) {
            last_servo_tick = now;
        }
        const bool scan_sync_marker = finish_sync_busy_.load(std::memory_order_acquire);
        const uint32_t num_pages = drain_all_devices(scan_sync_marker, page_buf, record_buf);
        num_pages_received += num_pages;
        if (scan_sync_marker || servo_ticked) {
            service_finish_sync(now, servo_ticked);
        }
#if defined(TRACY_ENABLE) && TT_TRACY_CATEGORY_RT_PROFILER
        // After the sync so the plots read fresh values.
        if (servo_ticked) {
            emit_diagnostics_plots();
        }
#endif
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
    // Wake consumers only when records were actually published, not merely when pages were drained. Sync-handshake
    // pages (SYNC_CHECK markers) are drained and consumed by the offset re-anchor but publish nothing, so waking on
    // page count would spuriously wake every consumer once per servo interval even with no records flowing.
    uint32_t num_pages = 0;
    size_t records_published = 0;
    for (auto& dev_state : devices_) {
        try {
            const DrainCounts counts = drain_device_pages(dev_state, scan_sync_marker, page_buf, record_buf);
            num_pages += counts.pages;
            records_published += counts.records;
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal, "[Real-time profiler] Exception draining device {}: {}", dev_state.chip_id, e.what());
        }
    }
    if (records_published > 0) {
        realtime_profiler_service_->wake_consumers();
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
    constexpr auto kShutdownKernelExitGrace = std::chrono::milliseconds(100);

    // Re-write ring_buffer->terminate as a safety net, then let the push kernel deliver the last PCIe page.
    for (auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer != 0 && dev_state.device) {
            const uint32_t terminate_addr = dev_state.core_l1.ring_buffer + offsetof(RtProfilerRingBuffer, terminate);
            std::vector<uint32_t> terminate_flag = {1};
            try {
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
        receiver_thread_.join();
    }

    if (ring_attached_) {
        realtime_profiler_service_->detach_ring(*ring_);
        ring_attached_ = false;
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

    devices_.clear();
}

void RealtimeProfilerManager::run_sync(DeviceState& dev_state, uint32_t num_samples) {
    constexpr auto kRunSyncSettleDelay = std::chrono::milliseconds(50);
    constexpr auto kRunSyncSampleInterval = std::chrono::milliseconds(5);
    constexpr auto kRunSyncReadTimeout = std::chrono::milliseconds(2000);
    constexpr uint32_t kRunSyncMaxConsecutiveTimeouts = 3;
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const int64_t host_start_time = realtime_profiler_host_timestamp();

    struct SyncSample {
        int64_t host_time;     // Full 64-bit host TSC ticks relative to host_start_time
        uint64_t device_time;  // Device wall clock cycles
    };
    std::vector<SyncSample> samples;

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

    std::this_thread::sleep_for(kRunSyncSettleDelay);

    uint32_t consecutive_timeouts = 0;

    for (uint32_t i = 0; i < num_samples + 1; i++) {
        std::this_thread::sleep_for(kRunSyncSampleInterval);

        // Send the truncated 32-bit host timestamp as the echo identifier the device pairs its capture against.
        const int64_t host_before = realtime_profiler_host_timestamp() - host_start_time;
        const uint32_t host_time_id = static_cast<uint32_t>(host_before);
        write_sync_timestamp(dev_state, host_time_id);

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

    // Mean-centered linear regression for slope (device cycles per TSC tick); centering avoids catastrophic
    // cancellation at absolute-timestamp magnitudes.
    if (samples.size() >= 2) {
        const double n = static_cast<double>(samples.size());
        const double host_ns_per_tick = realtime_profiler_host_ns_per_tick();

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
            dev_state.device_cycles_per_host_tick = num / den;
            dev_state.sync_frequency = dev_state.device_cycles_per_host_tick / host_ns_per_tick;
        } else {
            dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
            dev_state.device_cycles_per_host_tick = dev_state.sync_frequency * host_ns_per_tick;
        }

        // Intercept via means: intercept = ȳ - slope * x̄ = device cycle count at host_time = 0.
        const double intercept = device_mean - dev_state.device_cycles_per_host_tick * host_mean;
        dev_state.first_timestamp = static_cast<uint64_t>(intercept);
        dev_state.sync_host_start = host_start_time;

        // Residual of each handshake sample vs the fitted line, in host ns — the per-sample sync jitter.
        double residual_sumsq_ns = 0.0;
        double residual_max_ns = 0.0;
        for (const auto& s : samples) {
            const double predicted_device =
                device_mean + dev_state.device_cycles_per_host_tick * (static_cast<double>(s.host_time) - host_mean);
            const double residual_ns =
                (static_cast<double>(s.device_time) - predicted_device) / dev_state.sync_frequency;
            residual_sumsq_ns += residual_ns * residual_ns;
            residual_max_ns = std::max(residual_max_ns, std::abs(residual_ns));
        }
        const double residual_rms_ns = std::sqrt(residual_sumsq_ns / n);

        log_info(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync complete: {} samples, frequency={:.6f} GHz, "
            "device_time_at_sync={} cycles, fit residual rms={:.0f} ns max={:.0f} ns",
            dev_state.chip_id,
            samples.size(),
            dev_state.sync_frequency,
            dev_state.first_timestamp,
            residual_rms_ns,
            residual_max_ns);
    } else {
        dev_state.sync_frequency = cluster.get_device_aiclk(dev_state.chip_id) / 1000.0;
        dev_state.device_cycles_per_host_tick = dev_state.sync_frequency * realtime_profiler_host_ns_per_tick();
        dev_state.first_timestamp = 0;
        dev_state.sync_host_start = host_start_time;
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Device {} sync failed - not enough samples, using default frequency",
            dev_state.chip_id);
    }
    reanchor_device_cycle_offset(dev_state, dev_state.sync_host_start, dev_state.first_timestamp);
}

void RealtimeProfilerManager::reanchor_device_cycle_offset(
    DeviceState& dev_state, int64_t host_anchor, uint64_t device_anchor) {
    // Drift residual (host_real - host_predicted, pre-update); then move the offset, holding the fitted slope fixed.
    const double host_pred_ns =
        (static_cast<double>(device_anchor) - static_cast<double>(dev_state.device_cycle_offset)) /
        dev_state.sync_frequency;
    dev_state.sync_tracking_error_ns =
        std::llround(static_cast<double>(host_anchor) * realtime_profiler_host_ns_per_tick() - host_pred_ns);
    dev_state.device_cycle_offset = std::llround(
        static_cast<double>(device_anchor) - dev_state.device_cycles_per_host_tick * static_cast<double>(host_anchor));
}

int64_t RealtimeProfilerManager::measure_sync_rtt_ticks(
    const DeviceState& dev_state, int64_t host_before, uint32_t host_time_id) {
    if (dev_state.ack_host_ptr == nullptr) {
        return 0;
    }
    // Poll our own pinned host word, which the device NOC-writes the token into (bypassing the record FIFO). Skip the
    // deadline check for the first kRttProbeHealthyPolls reads so a healthy handshake (far fewer) never reads a clock
    // inside the round trip it is timing; the endpoint below is read fresh, so the measurement stays exact.
    const int64_t deadline =
        host_before + static_cast<int64_t>(kRttProbeTimeoutNs / realtime_profiler_host_ns_per_tick());
    uint32_t polls = 0;
    while (*dev_state.ack_host_ptr != host_time_id) {
        if (++polls > kRttProbeHealthyPolls && realtime_profiler_host_timestamp() > deadline) {
            return -1;  // ACK never observed: not a valid round trip, so the caller keeps the previous bound
        }
    }
    return realtime_profiler_host_timestamp() - host_before;
}

void RealtimeProfilerManager::cache_calibration(const DeviceState& dev_state) {
    std::lock_guard<std::mutex> lock(g_rt_profiler_init_sync_mu);
    auto& entry = g_rt_profiler_calibration_by_chip[dev_state.chip_id];
    entry.updated_at = std::chrono::steady_clock::now();
    entry.sync_frequency = dev_state.sync_frequency;
    entry.device_cycles_per_host_tick = dev_state.device_cycles_per_host_tick;
    entry.device_cycle_offset = dev_state.device_cycle_offset;
    entry.sync_rtt_ticks = dev_state.sync_rtt_ticks;
}

}  // namespace tt::tt_metal::distributed
