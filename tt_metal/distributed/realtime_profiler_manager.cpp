// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/realtime_profiler_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "llrt/hal.hpp"
#include "tools/profiler/tt_metal_tracy.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/dispatch/realtime_profiler_tracy_handler.hpp"
#include "jit_build/build_env_manager.hpp"
#include "tools/profiler/x280_driver.hpp"
#include "hostdevcommon/profiler_common.h"

namespace tt::tt_metal::distributed {

namespace {

// Minimum wall time between full init calibrations (run_sync + constructor SYNC_CHECK) and
// between finish-path sync checks, per physical chip. Matches the finish-path throttle.
constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

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
    static constexpr uint32_t fifo_size = 4096;                    // 4KB pinned-host FIFO for D2H socket
    static constexpr uint32_t page_size = RT_PROFILER_ENTRY_SIZE;  // host page size == ring entry size
    static constexpr uint32_t core_l1_size = sizeof(RealtimeProfilerCoreL1);
};

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
RealtimeProfilerManager::DeviceState::DeviceState(DeviceState&& o) noexcept :
    device(o.device),
    chip_id(o.chip_id),
    mesh_coord(std::move(o.mesh_coord)),
    realtime_profiler_core(o.realtime_profiler_core),
    socket(std::move(o.socket)),
    x280_socket(std::move(o.x280_socket)),
    x280_driver(std::move(o.x280_driver)),
    x280_params_addr(o.x280_params_addr),
    x280_active(o.x280_active),
    x280_virt_to_noc0(std::move(o.x280_virt_to_noc0)),
    realtime_profiler_program(std::move(o.realtime_profiler_program)),
    core_l1(o.core_l1),
    first_timestamp(o.first_timestamp),
    sync_host_start(o.sync_host_start),
    sync_frequency(o.sync_frequency),
    realtime_profiler_base_addr(o.realtime_profiler_base_addr),
    sync_request_addr(o.sync_request_addr),
    sync_host_ts_addr(o.sync_host_ts_addr),
    sync_response_received(o.sync_response_received.load(std::memory_order_relaxed)),
    sync_host_time_before(o.sync_host_time_before),
    last_finish_sync_at(o.last_finish_sync_at),
    pending_first_unthrottled_finish_sync(o.pending_first_unthrottled_finish_sync) {}

RealtimeProfilerManager::RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device) :
    context_id_(mesh_device->impl().get_context_id()) {
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

        dev_state.realtime_profiler_base_addr = realtime_profiler_base_addr;
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
            uint32_t profiler_msg_carve_base = realtime_profiler_base_addr;

            std::vector<uint32_t> noc_xy_data = {realtime_profiler_noc_xy};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                profiler_msg_carve_base + realtime_profiler_core_noc_xy_offset,
                noc_xy_data,
                CoreType::WORKER);

            std::vector<uint32_t> remote_state_addr_data = {realtime_profiler_core_state_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                profiler_msg_carve_base + remote_state_addr_field_offset,
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
            if (x280_cluster.arch() == tt::ARCH::BLACKHOLE &&
                !soc.get_cores(CoreType::L2CPU, CoordSystem::NOC0).empty() && !x280_fw.empty()) {
                constexpr int kL2CpuIndex = 0;  // tile (8,3) — proven single-chip path
                constexpr int kX280PllMhz = 1000;
                constexpr uint32_t kX280ConfigAddr = 0x08019000u;  // X280 LIM: above STAGECTL, below STAGE_BASE
                constexpr uint64_t kX280MboxParams = 0x08011000ull;
                constexpr uint64_t kX280MboxResults = 0x08011040ull;
                constexpr uint64_t kX280MboxCoords = 0x08011200ull;
                constexpr uint32_t kX280Fifo = 4096;
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

                std::vector<uint8_t> params(64, 0), results(64, 0);
                auto pk = [&](size_t off, uint64_t val) { std::memcpy(params.data() + off, &val, 8); };
                pk(0x00, kX280ConfigAddr);
                pk(0x08, static_cast<uint64_t>(pc.x));
                pk(0x10, static_cast<uint64_t>(pc.y));
                pk(0x18, prof_l1);
                pk(0x20, num_cores);
                pk(0x28, 0);  // P_STOP = 0: run continuously until shutdown
                zx.write_block(params.data(), (uint32_t)params.size(), kX280MboxParams);
                zx.write_block(results.data(), (uint32_t)results.size(), kX280MboxResults);
                dev_state.x280_params_addr = kX280MboxParams;

                zx.set_reset_vectors(profiler::X280_LIM_BASE);
                zx.set_pll(kX280PllMhz);
                zx.release_reset();

                // Fast-fail liveness check: poll profzone's main()-entry heartbeat (RES @ params+0x30
                // == params+0x70) for a few ms. If the core never writes it, the X280 isn't executing
                // — on a fresh board this is almost always because the L3 LIM ECC was never primed, so
                // profzone's stores fault silently. Rather than leave a half-booted drainer that never
                // drains (and lets a filling SPSC ring deadlock a producing RISC), skip it here with an
                // actionable message. One-time per power cycle: `tt x280-prime <target>` then rerun.
                constexpr uint64_t kX280HbMainMagic = 0xB007ULL;
                uint64_t hb = 0;
                for (int i = 0; i < 300 && hb != kX280HbMainMagic; i++) {
                    hb = zx.lim_rd_u64(dev_state.x280_params_addr + 0x70);
                    if (hb != kX280HbMainMagic) {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                }
                if (hb != kX280HbMainMagic) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Device {}: X280 drainer FW did not start (l2cpu {}, "
                        "heartbeat=0x{:x}). The L2CPU's L3 LIM ECC is likely not primed on this board — "
                        "run `tt x280-prime {}` once after each cold power cycle, then rerun. Continuing "
                        "without X280 kernel-zone capture.",
                        device_id,
                        kL2CpuIndex,
                        hb,
                        device_id);
                    zx.assert_reset();
                    dev_state.x280_socket.reset();
                    dev_state.x280_driver.reset();
                    dev_state.x280_active = false;
                } else {
                    dev_state.x280_active = true;
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
        }

        MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
        devices_.push_back(std::move(dev_state));
    }

    if (devices_.empty()) {
        log_debug(
            tt::LogMetal, "[Real-time profiler] No local devices found in mesh, skipping real-time profiler setup");
        return;
    }

    // Announce activation; paired with NotifyProgramRealtimeProfilerDeactivated on shutdown.
    for (const auto& dev_state : devices_) {
        tt::NotifyProgramRealtimeProfilerActivated(dev_state.chip_id);
    }

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
        constexpr uint32_t kMaxSyncRetries = 3;
        constexpr uint32_t kRetryDelayMs = 500;
        for (uint32_t attempt = 0; attempt <= kMaxSyncRetries; attempt++) {
            if (attempt > 0) {
                log_debug(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} sync retry {}/{}",
                    dev_state.chip_id,
                    attempt,
                    kMaxSyncRetries);
                std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDelayMs));
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
        std::vector<uint32_t> sync_req = {1};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_request_addr,
            sync_req,
            CoreType::WORKER);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

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

        constexpr uint32_t kSyncCheckTimeoutMs = 3000;
        auto sc_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kSyncCheckTimeoutMs);
        bool sc_got_response = false;
        while (std::chrono::steady_clock::now() < sc_deadline) {
            if (dev_state.socket->pages_available() > 0) {
                sc_got_response = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        sync_req[0] = 0;
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_request_addr,
            sync_req,
            CoreType::WORKER);

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
                kSyncCheckTimeoutMs);
        }
    });

    for (auto& dev_state : devices_) {
        dev_state.pending_first_unthrottled_finish_sync = true;
    }

    // Background receiver thread that polls all device sockets round-robin.
    stop_.store(false);
    receiver_thread_ = std::thread([this]() {
        tracy::SetThreadName("RealtimeProfiler");
        uint64_t pages_received = 0;
        uint64_t x280_pages = 0;

        log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

        // Process one page from a device socket. Returns true if a page was consumed.
        std::vector<uint32_t> page_buf(RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t));
        auto process_one_page = [&](DeviceState& dev_state) -> bool {
            uint32_t available = dev_state.socket->pages_available();
            if (available == 0) {
                return false;
            }

            // TODO: Uncomment this and apply a debug verbosity level when
            // https://github.com/tenstorrent/tt-metal/issues/30615 is done.
            // ZoneScopedN("ProcessPage");
            dev_state.socket->read(page_buf.data(), 1);
            uint32_t* read_ptr = page_buf.data();

            uint32_t marker = read_ptr[3];
            if (!dev_state.sync_response_received.load() && marker == REALTIME_PROFILER_SYNC_MARKER_ID) {
                uint64_t device_time = (static_cast<uint64_t>(read_ptr[0]) << 32) | read_ptr[1];
                tracy_handler_->CalibrateDevice(
                    dev_state.chip_id, dev_state.sync_host_time_before, device_time, dev_state.sync_frequency);
                tracy_handler_->PushSyncCheckMarker(dev_state.chip_id, device_time, dev_state.sync_frequency);
                pages_received++;
                dev_state.sync_response_received.store(true);
                return true;
            }

            // kernel_start (words 0-3), kernel_end (words 4-7); each
            // realtime_profiler_timestamp_t: time_hi, time_lo, id, header.
            uint64_t start_time = (static_cast<uint64_t>(read_ptr[0]) << 32) | read_ptr[1];
            uint32_t start_id = read_ptr[2];
            uint64_t end_time = (static_cast<uint64_t>(read_ptr[4]) << 32) | read_ptr[5];

            // Skip records with id==0 (non-GO dispatch commands like SET_NUM_WORKER_SEMS):
            // they have no valid program and may carry stale end timestamps.
            if (start_id != 0) {
                // TODO: Uncomment this and apply a debug verbosity level when
                // https://github.com/tenstorrent/tt-metal/issues/30615 is done.
                // ZoneScopedN("InvokeCallbacks");
                tt::ProgramRealtimeRecord record{
                    .runtime_id = start_id,
                    .chip_id = dev_state.chip_id,
                    .start_timestamp = start_time,
                    .end_timestamp = end_time,
                    .frequency = dev_state.sync_frequency,
                    .kernel_sources = tt::GetKernelSourcesForRuntimeId(static_cast<uint16_t>(start_id)),
                };
                tt::InvokeProgramRealtimeProfilerCallbacks(record);
            }

            pages_received++;
            return true;
        };

        // Process one X280 raw-marker page (64 B, first 24 used): [0]=core_x(virtual) [1]=core_y(virtual)
        // [2]=risc [3]=timer_id(type|hash) [4]=time_hi [5]=time_lo. Relayed in ring order (== emission
        // order == correct nest order per lane), so pushing START/END per marker in arrival order lets
        // Tracy nest them correctly. Coords are translated virtual->NOC0 to match the standard profiler.
        std::vector<uint32_t> x280_page_buf(RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t));
        // 16-bit-hash -> zone name/file/line, resolved the same way the DRAM profiler names zones.
        // Built lazily on the first marker (kernels have compiled + written the zone-source logs by
        // the time any marker is drained), so device zones show real names instead of "Zone_<hash>".
        std::unordered_map<uint16_t, tracy::MarkerDetails> x280_zone_names;
        std::once_flag x280_zone_names_once;
        auto process_x280_page = [&](DeviceState& dev_state) -> bool {
            if (!dev_state.x280_active || !dev_state.x280_socket) {
                return false;
            }
            if (dev_state.x280_socket->pages_available() == 0) {
                return false;
            }
            dev_state.x280_socket->read(x280_page_buf.data(), 1);
            const uint32_t* p = x280_page_buf.data();
            const uint32_t vx = p[0], vy = p[1], risc = p[2], timer_id = p[3];
            const uint64_t timestamp = (static_cast<uint64_t>(p[4]) << 32) | p[5];
            // virtual -> NOC0 (built at boot); fall back to the raw virtual coord if unmapped.
            uint32_t core_x = vx, core_y = vy;
            if (auto it = dev_state.x280_virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy);
                it != dev_state.x280_virt_to_noc0.end()) {
                core_x = it->second.first;
                core_y = it->second.second;
            }
            // Resolve the zone name/file/line from the 16-bit hash (same map the DRAM profiler uses).
            std::call_once(x280_zone_names_once, [&] {
                try {
                    x280_zone_names = loadZoneSourceLocationsHashesReadOnly();
                } catch (const std::exception& e) {
                    log_warning(tt::LogMetal, "[Real-time profiler] X280 zone-name resolution failed: {}", e.what());
                }
                log_debug(
                    tt::LogMetal, "[Real-time profiler] X280 resolved {} zone-name hashes", x280_zone_names.size());
            });
            const tracy::MarkerDetails* details = nullptr;
            if (auto it = x280_zone_names.find(static_cast<uint16_t>(timer_id & 0xFFFF)); it != x280_zone_names.end()) {
                details = &it->second;
            }
            // Debug/comparison (env TT_METAL_X280_ZONE_CSV=<path>): dump every relayed marker so the
            // X280 capture can be diffed 1:1 against the DRAM profiler's profile_log_device.csv by
            // marker identity. timer_id low-16 = zone-name hash and coords are NOC0, both matching the
            // DRAM CSV. ptype: 0=ZONE_START, 1=ZONE_END. Off by default.
            static std::ofstream x280_zone_csv;
            static std::once_flag x280_zone_csv_once;
            std::call_once(x280_zone_csv_once, [] {
                if (const char* pth = std::getenv("TT_METAL_X280_ZONE_CSV"); pth != nullptr && *pth != '\0') {
                    x280_zone_csv.open(pth);
                    x280_zone_csv << "chip,core_x,core_y,risc,timer_id,ptype,cycle,name\n";
                }
            });
            if (x280_zone_csv.is_open()) {
                x280_zone_csv << dev_state.chip_id << ',' << core_x << ',' << core_y << ',' << risc << ','
                              << (timer_id & 0xFFFF) << ',' << ((timer_id >> 16) & 0x7) << ',' << timestamp << ','
                              << (details != nullptr ? details->marker_name : "") << '\n';
            }
            tracy_handler_->PushDeviceMarker(dev_state.chip_id, core_x, core_y, risc, timer_id, timestamp, details);
            pages_received++;
            x280_pages++;
            return true;
        };

        while (!stop_.load()) {
            if (pause_requested_.load(std::memory_order_acquire)) {
                paused_.store(true, std::memory_order_release);
                while (pause_requested_.load(std::memory_order_acquire) && !stop_.load()) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                paused_.store(false, std::memory_order_release);
                continue;
            }

            // TODO: Uncomment this and apply a debug verbosity level when
            // https://github.com/tenstorrent/tt-metal/issues/30615 is done.
            // ZoneScopedN("PollLoop");
            bool any_data = false;

            for (auto& dev_state : devices_) {
                try {
                    if (process_one_page(dev_state)) {
                        any_data = true;
                    }
                    if (process_x280_page(dev_state)) {
                        any_data = true;
                    }
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "[Real-time profiler] Exception in receiver for device {}: {}",
                        dev_state.chip_id,
                        e.what());
                }
            }

            if (!any_data) {
                // TODO: Uncomment this and apply a debug verbosity level when
                // https://github.com/tenstorrent/tt-metal/issues/30615 is done.
                // ZoneScopedN("Idle");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Drain in-flight PCIe pages until all sockets stay empty for several rounds.
        {
            // TODO: Uncomment this and apply a debug verbosity level when
            // https://github.com/tenstorrent/tt-metal/issues/30615 is done.
            // ZoneScopedN("DrainShutdown");
            constexpr uint32_t kDrainQuietRounds = 10;
            uint64_t drain_pages = 0;
            uint32_t quiet_rounds = 0;
            while (quiet_rounds < kDrainQuietRounds) {
                bool any_data = false;
                for (auto& dev_state : devices_) {
                    try {
                        if (process_one_page(dev_state)) {
                            any_data = true;
                            drain_pages++;
                        }
                        if (process_x280_page(dev_state)) {
                            any_data = true;
                            drain_pages++;
                        }
                    } catch (const std::exception& e) {
                        log_warning(
                            tt::LogMetal,
                            "[Real-time profiler] Exception draining device {}: {}",
                            dev_state.chip_id,
                            e.what());
                    }
                }
                if (any_data) {
                    quiet_rounds = 0;
                } else {
                    quiet_rounds++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            log_info(
                tt::LogMetal,
                "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown); "
                "{} were X280 kernel-zone pages",
                pages_received,
                drain_pages,
                x280_pages);
        }
    });
}

RealtimeProfilerManager::~RealtimeProfilerManager() { shutdown(); }

void RealtimeProfilerManager::shutdown() {
    // Enter the profiler TERMINATE phase FIRST: broadcast PROFILER_TERMINATE=1 into every profiled
    // core's control vector so a producing RISC blocked on a full SPSC ring (in ring_ensure_room)
    // stops waiting and proceeds. The X280 drainer is about to stop; without this a core with a
    // full ring stays blocked and close_devices()->wait_until_cores_done() spins forever. Must run
    // before the X280 is P_STOP'd/reset below.
    {
        auto& mctx = MetalContext::instance(context_id_);
        auto& cluster = mctx.get_cluster();
        const uint64_t prof_l1 = mctx.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
        const uint64_t terminate_addr =
            prof_l1 + static_cast<uint64_t>(kernel_profiler::PROFILER_TERMINATE) * sizeof(uint32_t);
        const uint32_t one = 1;
        for (auto& dev_state : devices_) {
            if (!dev_state.device) {
                continue;
            }
            try {
                // FULL logical worker grid (not just compute_with_storage) so DISPATCH cores —
                // which are profiled when PROFILER_OPT_DO_DISPATCH_CORES is set and are the ones
                // wait_for_dispatch_cores() waits on — also get the terminate flag.
                CoreCoord grid = dev_state.device->logical_grid_size();
                for (uint32_t ly = 0; ly < static_cast<uint32_t>(grid.y); ly++) {
                    for (uint32_t lx = 0; lx < static_cast<uint32_t>(grid.x); lx++) {
                        CoreCoord v = cluster.get_virtual_coordinate_from_logical_coordinates(
                            dev_state.chip_id, CoreCoord{lx, ly}, CoreType::WORKER);
                        cluster.write_core(&one, sizeof(one), tt_cxy_pair(dev_state.chip_id, v), terminate_addr);
                    }
                }
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: failed to broadcast PROFILER_TERMINATE: {}",
                    dev_state.chip_id,
                    e.what());
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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (receiver_thread_.joinable()) {
        stop_.store(true);
        receiver_thread_.join();
    }

    // Park each X280 in reset now that its socket has been drained.
    for (auto& dev_state : devices_) {
        if (dev_state.x280_active && dev_state.x280_driver) {
            try {
                // Telemetry: profzone's result mailbox (results base = params_addr + 0x40).
                // total_markers@+0x00 = raw markers relayed; loops@+0x08 = drain-loop passes.
                uint64_t total_markers = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x40);
                uint64_t loops = dev_state.x280_driver->lim_rd_u64(dev_state.x280_params_addr + 0x48);
                log_info(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: X280 drainer relayed {} markers ({} drain passes)",
                    dev_state.chip_id,
                    total_markers,
                    loops);
                dev_state.x280_driver->assert_reset();
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Failed to reset X280 on device {}: {}",
                    dev_state.chip_id,
                    e.what());
            }
        }
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

    std::vector<uint32_t> sync_req_data = {1};
    tt::tt_metal::detail::WriteToDeviceL1(
        dev_state.device,
        dev_state.realtime_profiler_core,
        dev_state.sync_request_addr,
        sync_req_data,
        CoreType::WORKER);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    constexpr uint32_t kSyncReadTimeoutMs = 2000;
    uint32_t consecutive_timeouts = 0;
    constexpr uint32_t kMaxConsecutiveTimeouts = 3;

    for (uint32_t i = 0; i < num_samples + 1; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

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

        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kSyncReadTimeoutMs);
        bool got_response = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (dev_state.socket->pages_available() > 0) {
                got_response = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
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
                kSyncReadTimeoutMs,
                consecutive_timeouts,
                kMaxConsecutiveTimeouts);
            if (consecutive_timeouts >= kMaxConsecutiveTimeouts) {
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
            samples.push_back({host_before, device_time});
        }
    }

    sync_req_data[0] = 0;
    tt::tt_metal::detail::WriteToDeviceL1(
        dev_state.device,
        dev_state.realtime_profiler_core,
        dev_state.sync_request_addr,
        sync_req_data,
        CoreType::WORKER);

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

void RealtimeProfilerManager::trigger_sync_check() {
    if (devices_.empty() || !tracy_handler_) {
        return;
    }

    constexpr uint32_t kPageSize = 64;
    constexpr uint32_t kPageWords = kPageSize / sizeof(uint32_t);
    constexpr uint32_t kSyncTimeoutMs = 5000;
    constexpr uint32_t kPauseTimeoutMs = 2000;

    const auto throttle_now = std::chrono::steady_clock::now();
    std::vector<size_t> device_indices_to_sync;
    device_indices_to_sync.reserve(devices_.size());
    for (size_t i = 0; i < devices_.size(); i++) {
        const auto& dev_state = devices_[i];
        const bool interval_elapsed = !dev_state.last_finish_sync_at.has_value() ||
                                      throttle_now - *dev_state.last_finish_sync_at >= kRtProfilerMinSyncInterval;
        if (interval_elapsed || dev_state.pending_first_unthrottled_finish_sync) {
            device_indices_to_sync.push_back(i);
        }
    }
    if (device_indices_to_sync.empty()) {
        return;
    }

    // 1. Pause the receiver thread for exclusive socket access. This breaks a potential
    //    GIL deadlock: the receiver may be waiting on the GIL for a Python callback while
    //    the caller (holding the GIL) blocks here. Skip the check if pause times out.
    pause_requested_.store(true, std::memory_order_release);
    {
        auto pause_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kPauseTimeoutMs);
        while (!paused_.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() > pause_deadline) {
                log_warning(
                    tt::LogMetal, "[Real-time profiler] Could not pause receiver thread for sync check - skipping");
                pause_requested_.store(false, std::memory_order_release);
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Only devices in device_indices_to_sync enter sync mode and emit FINISH_SYNC; others
    // keep receiving via the receiver thread once it resumes (no device-side sync_request).
    parallel_for_each_device_index(device_indices_to_sync, [&](size_t dev_index) {
        auto& dev_state = devices_[dev_index];
        std::vector<uint32_t> page_buf(kPageWords);

        // 2. Drain pending data pages so the socket has room for the sync response. Each
        //    page is processed the same way the receiver thread would.
        while (dev_state.socket->pages_available() > 0) {
            dev_state.socket->read(page_buf.data(), 1);
            uint32_t* rp = page_buf.data();
            uint64_t start_time = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
            uint32_t start_id = rp[2];
            uint64_t end_time = (static_cast<uint64_t>(rp[4]) << 32) | rp[5];
            if (start_id != 0) {
                tt::ProgramRealtimeRecord record{
                    .runtime_id = start_id,
                    .chip_id = dev_state.chip_id,
                    .start_timestamp = start_time,
                    .end_timestamp = end_time,
                    .frequency = dev_state.sync_frequency,
                    .kernel_sources = tt::GetKernelSourcesForRuntimeId(static_cast<uint16_t>(start_id)),
                };
                std::lock_guard<std::mutex> cb_lock(parallel_finish_sync_callback_mu_);
                tt::InvokeProgramRealtimeProfilerCallbacks(record);
            }
        }

        // 3. Enter sync mode on the device kernel.
        std::vector<uint32_t> sync_req = {1};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_request_addr,
            sync_req,
            CoreType::WORKER);

        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // 4. Send host timestamp to trigger device response. Tracy marker goes immediately
        //    before the PCIe write so FINISH_SYNC and SYNC_CHECK share a timing convention.
        dev_state.sync_host_time_before = rt_profiler_host_ticks();
        uint32_t host_time_id = static_cast<uint32_t>(dev_state.sync_host_time_before & 0xFFFFFFFF);
        std::vector<uint32_t> host_time_data = {host_time_id};
        TracyMessageL("FINISH_SYNC");
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_host_ts_addr,
            host_time_data,
            CoreType::WORKER);

        // 5. Read until the sync response arrives or we time out; data pages that arrive
        //    in the meantime are processed inline.
        bool got_sync = false;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kSyncTimeoutMs);
        while (!got_sync) {
            if (std::chrono::steady_clock::now() > deadline) {
                log_warning(tt::LogMetal, "[Real-time profiler] Sync check timed out for device {}", dev_state.chip_id);
                break;
            }

            if (dev_state.socket->pages_available() == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            dev_state.socket->read(page_buf.data(), 1);
            uint32_t* rp = page_buf.data();
            uint32_t marker = rp[3];

            if (marker == REALTIME_PROFILER_SYNC_MARKER_ID) {
                uint64_t device_time = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
                tracy_handler_->CalibrateDevice(
                    dev_state.chip_id, dev_state.sync_host_time_before, device_time, dev_state.sync_frequency);
                tracy_handler_->PushSyncCheckMarker(dev_state.chip_id, device_time, dev_state.sync_frequency);
                dev_state.last_finish_sync_at = std::chrono::steady_clock::now();
                dev_state.pending_first_unthrottled_finish_sync = false;
                got_sync = true;
            } else {
                uint64_t start_time = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
                uint32_t start_id = rp[2];
                uint64_t end_time = (static_cast<uint64_t>(rp[4]) << 32) | rp[5];
                if (start_id != 0) {
                    tt::ProgramRealtimeRecord record{
                        .runtime_id = start_id,
                        .chip_id = dev_state.chip_id,
                        .start_timestamp = start_time,
                        .end_timestamp = end_time,
                        .frequency = dev_state.sync_frequency,
                        .kernel_sources = tt::GetKernelSourcesForRuntimeId(static_cast<uint16_t>(start_id)),
                    };
                    std::lock_guard<std::mutex> cb_lock(parallel_finish_sync_callback_mu_);
                    tt::InvokeProgramRealtimeProfilerCallbacks(record);
                }
            }
        }

        // 6. Exit sync mode.
        sync_req[0] = 0;
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_request_addr,
            sync_req,
            CoreType::WORKER);
    });

    // 7. Resume receiver thread.
    pause_requested_.store(false, std::memory_order_release);
}

D2HSocket* RealtimeProfilerManager::get_socket() const {
    return devices_.empty() ? nullptr : devices_.front().socket.get();
}

}  // namespace tt::tt_metal::distributed
