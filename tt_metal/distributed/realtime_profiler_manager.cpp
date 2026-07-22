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
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/dispatch/realtime_profiler_tracy_handler.hpp"
#include "tt_metal/impl/profiler/profiler.hpp"                // tt::tt_metal::SyncInfo, DeviceProfiler
#include "tt_metal/impl/profiler/profiler_state_manager.hpp"  // ProfilerStateManager

namespace tt::tt_metal::distributed {

namespace {

// Minimum wall time between full init calibrations (run_sync + constructor SYNC_CHECK) and
// between finish-path sync checks, per physical chip. Matches the finish-path throttle.
constexpr auto kRtProfilerMinSyncInterval = std::chrono::seconds(60);

// Last full init sync per chip, process-wide, to avoid repeating ~0.5s run_sync on every mesh open.
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
        std::vector<uint32_t> sync_req = {1};
        tt::tt_metal::detail::WriteToDeviceL1(
            dev_state.device,
            dev_state.realtime_profiler_core,
            dev_state.sync_request_addr,
            sync_req,
            CoreType::WORKER);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

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

        log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

        // Process one page from a device socket. Returns true if a page was consumed.
        std::vector<uint32_t> page_buf(RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t));
        auto process_one_page = [&](DeviceState& dev_state) -> bool {
            uint32_t available = dev_state.socket->pages_available();
            if (available == 0) {
                return false;
            }

            TTZoneScopedDN(RT_PROFILER, "ProcessPage");
            dev_state.socket->read(page_buf.data(), 1);
            uint32_t* read_ptr = page_buf.data();

            uint32_t marker = read_ptr[3];
            if (!dev_state.sync_response_received.load() && marker == REALTIME_PROFILER_SYNC_MARKER_ID) {
                uint64_t device_time = (static_cast<uint64_t>(read_ptr[0]) << 32) | read_ptr[1];
                tracy_handler_->CalibrateDevice(
                    dev_state.chip_id, dev_state.sync_host_time_before, device_time, dev_state.sync_frequency);
                tracy_handler_->PushSyncCheckMarker(dev_state.chip_id, device_time, dev_state.sync_frequency);
                publish_device_profiler_sync_anchor(
                    dev_state.chip_id,
                    static_cast<double>(dev_state.sync_host_time_before),
                    static_cast<double>(device_time),
                    dev_state.sync_frequency,
                    dev_state.realtime_profiler_core.str());
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
                TTZoneScopedDN(RT_PROFILER, "InvokeCallbacks");
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

        while (!stop_.load()) {
            if (pause_requested_.load(std::memory_order_acquire)) {
                paused_.store(true, std::memory_order_release);
                while (pause_requested_.load(std::memory_order_acquire) && !stop_.load()) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                paused_.store(false, std::memory_order_release);
                continue;
            }

            TTZoneScopedDN(RT_PROFILER, "PollLoop");
            bool any_data = false;

            for (auto& dev_state : devices_) {
                try {
                    if (process_one_page(dev_state)) {
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
                TTZoneScopedDN(RT_PROFILER, "Idle");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Drain in-flight PCIe pages until all sockets stay empty for several rounds.
        {
            TTZoneScopedDN(RT_PROFILER, "DrainShutdown");
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

            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown)",
                pages_received,
                drain_pages);
        }
    });
}

RealtimeProfilerManager::~RealtimeProfilerManager() { shutdown(); }

void RealtimeProfilerManager::shutdown() {
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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (receiver_thread_.joinable()) {
        stop_.store(true);
        receiver_thread_.join();
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

        log_info(
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
    std::scoped_lock map_lock(psm->device_profiler_map_mutex);
    psm->device_profiler_map.at(chip_id).realtime_sync_line =
        tt::tt_metal::DeviceProfiler::RealtimeSyncLine{host_anchor, device_anchor, frequency};
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

    // 1. Pause the receiver for exclusive socket access (breaks a GIL deadlock vs Python callbacks); skip if pause
    // times out.
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

    // Only these devices enter sync mode and emit FINISH_SYNC; others keep receiving once the thread resumes.
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
                publish_device_profiler_sync_anchor(
                    dev_state.chip_id,
                    static_cast<double>(dev_state.sync_host_time_before),
                    static_cast<double>(device_time),
                    dev_state.sync_frequency,
                    dev_state.realtime_profiler_core.str());
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
