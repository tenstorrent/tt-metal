// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"
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

    // The two profilers have separate data paths (this one reads dispatch_s timestamps through a reserved tensix),
    // but running them together produced ~1800 "zone with end < start" real-time records per ResNet run; until that
    // is understood the real-time profiler stands down in streaming mode.
    if (metal.rtoptions().get_streaming_profiler_enabled()) {
        log_info(
            tt::LogMetal, "Real-time profiler disabled on device {}: TT_METAL_STREAMING_PROFILER is set.", device_id);
        return {};
    }

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

}  // namespace

RealtimeProfilerManager::DeviceState::DeviceState() = default;
RealtimeProfilerManager::DeviceState::~DeviceState() = default;
RealtimeProfilerManager::DeviceState::DeviceState(DeviceState&&) noexcept = default;

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
        RT_PROFILER_SOCKET_CONFIG_SIZE >= D2HSocket::required_config_buffer_size(hal.get_alignment(HalMemType::L1)),
        "RT_PROFILER_SOCKET_CONFIG_SIZE ({} B) is smaller than D2HSocket's required config "
        "buffer size ({} B). Bump RT_PROFILER_SOCKET_CONFIG_SIZE in "
        "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp and rebuild.",
        RT_PROFILER_SOCKET_CONFIG_SIZE,
        D2HSocket::required_config_buffer_size(hal.get_alignment(HalMemType::L1)));
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

        IDevice* device = internal::get_device(*mesh_device, coord);
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
            MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
            continue;
        }

        dev_state.sync_request_addr = realtime_profiler_base_addr + sync_request_offset;
        dev_state.sync_host_ts_addr = realtime_profiler_base_addr + sync_host_timestamp_offset;

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
            dev_state.realtime_profiler_program = std::make_unique<Program>(CreateProgram(*mesh_device));
            auto& realtime_profiler_program = *dev_state.realtime_profiler_program;

            uint32_t dispatch_core_noc_x = 0;
            uint32_t dispatch_core_noc_y = 0;
            // Both carve-outs sit at realtime_profiler_base_addr, on dispatch_s and on the profiler core.
            const uint32_t dispatch_records_addr =
                realtime_profiler_base_addr + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                                                  realtime_profiler_msgs::realtime_profiler_msg_t::Field::records);
            const uint32_t dispatch_record_rd_idx_addr =
                realtime_profiler_base_addr +
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::record_rd_idx);
            if (dispatch_core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
                const tt_cxy_pair& dispatch_s_cxy = dispatch_core_manager.dispatcher_s_core(device_id, 0, 0);
                CoreCoord dispatch_s_virtual = device->virtual_core_from_logical_core(
                    CoreCoord(dispatch_s_cxy.x, dispatch_s_cxy.y), CoreType::WORKER);
                dispatch_core_noc_x = dispatch_s_virtual.x;
                dispatch_core_noc_y = dispatch_s_virtual.y;
            }

            DataMovementConfig brisc_config;
            brisc_config.processor = DataMovementProcessor::RISCV_0;
            brisc_config.noc = NOC::RISCV_0_default;
            brisc_config.defines["DISPATCH_CORE_NOC_X"] = std::to_string(dispatch_core_noc_x);
            brisc_config.defines["DISPATCH_CORE_NOC_Y"] = std::to_string(dispatch_core_noc_y);
            brisc_config.defines["DISPATCH_RECORDS_ADDR"] = std::to_string(dispatch_records_addr);
            brisc_config.defines["DISPATCH_RECORD_RD_IDX_ADDR"] = std::to_string(dispatch_record_rd_idx_addr);
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

            realtime_profiler_program.impl().compile(device, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::slow_dispatch::WriteRuntimeArgsToDevice(
                *device, realtime_profiler_program, /*force_slow_dispatch=*/true);
            ::tt::tt_metal::slow_dispatch::LaunchProgramAsync(
                *device, realtime_profiler_program, /*force_slow_dispatch=*/true);

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

        // Enable dispatch_s only now that the BRISC reader is running: dispatch_s waits for the reader to free
        // record slots, so enabling it first would stall dispatch if the launch above failed.
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
            uint32_t remote_wr_idx_addr_field_offset =
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_remote_wr_idx_addr);
            uint32_t record_wr_idx_offset = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                realtime_profiler_msgs::realtime_profiler_msg_t::Field::record_wr_idx);
            uint32_t realtime_profiler_core_wr_idx_addr = realtime_profiler_base_addr + record_wr_idx_offset;

            // Address first: dispatch_s starts publishing as soon as it sees a nonzero noc_xy.
            std::vector<uint32_t> remote_wr_idx_addr_data = {realtime_profiler_core_wr_idx_addr};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                realtime_profiler_base_addr + remote_wr_idx_addr_field_offset,
                remote_wr_idx_addr_data,
                CoreType::WORKER);

            std::vector<uint32_t> noc_xy_data = {realtime_profiler_noc_xy};
            tt::tt_metal::detail::WriteToDeviceL1(
                device,
                dispatch_s_core,
                realtime_profiler_base_addr + realtime_profiler_core_noc_xy_offset,
                noc_xy_data,
                CoreType::WORKER);

            dev_state.dispatch_s_core = dispatch_s_core;
            dev_state.record_full_wait_count_addr =
                realtime_profiler_base_addr +
                factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                    realtime_profiler_msgs::realtime_profiler_msg_t::Field::record_full_wait_count);

            log_debug(
                tt::LogMetal,
                "[Real-time profiler] Device {}: wrote real-time profiler core info (noc_xy=0x{:x}, "
                "remote_wr_idx_addr=0x{:x}) "
                "to dispatch_s ({}, {})",
                device_id,
                realtime_profiler_noc_xy,
                realtime_profiler_core_wr_idx_addr,
                dispatch_s_core.x,
                dispatch_s_core.y);
        }

        MetalContext::instance(context_id_).device_manager()->mark_rt_profiler_device_init_complete(device_id);
        devices_.push_back(std::move(dev_state));
    }
}

}  // namespace tt::tt_metal::distributed
