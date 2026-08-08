// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_metal.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {

inline RealtimeProfilerCoreL1Addrs compute_rt_profiler_core_l1_addrs(uint32_t base) {
    return {
        .ring_buffer = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, ring)),
        .socket_config = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, socket_config)),
    };
}

std::unique_ptr<distributed::D2HSocket> create_d2h_socket(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCoreCoord sender_core,
    uint32_t socket_config_l1_addr,
    ChipId device_id) {
    std::unique_ptr<distributed::D2HSocket> socket;
    try {
        socket = std::make_unique<distributed::D2HSocket>(
            mesh_device,
            sender_core,
            RealtimeProfilerRuntimeSizes::fifo_size,
            distributed::D2HSocket::ExternalConfigBuffer{.address = socket_config_l1_addr},
            distributed::D2HSocket::ProcessScope::InProcess);
        socket->set_page_size(RealtimeProfilerRuntimeSizes::page_size);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "Real-time profiler disabled on device {}: D2H socket construction failed ({}). "
            "This typically indicates a host-side memory pinning / hugepage mapping issue "
            "(e.g. IOMMU misconfiguration or UMD DMA pin failure). Continuing without RT "
            "profiler on this device.",
            device_id,
            e.what());
        return nullptr;
    }
    return socket;
}

// Tells dispatch_s where the profiler core is, so it can signal that core on program boundaries.
void publish_profiler_core_to_dispatch_s(
    IDevice* device,
    const Hal& hal,
    dispatch_core_manager& core_manager,
    CoreCoord profiler_core,
    uint32_t noc_xy_field_addr,
    uint32_t remote_state_field_addr,
    uint32_t profiler_state_addr) {
    const ChipId device_id = device->id();
    if (!core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
        return;
    }
    const tt_cxy_pair& dispatch_s_cxy = core_manager.dispatcher_s_core(device_id, 0, 0);
    const CoreCoord dispatch_s_core(dispatch_s_cxy.x, dispatch_s_cxy.y);
    const CoreCoord profiler_virtual = device->virtual_core_from_logical_core(profiler_core, CoreType::WORKER);

    const uint32_t profiler_noc_xy = hal.noc_xy_encoding(profiler_virtual.x, profiler_virtual.y);
    std::vector<uint32_t> noc_xy_data = {profiler_noc_xy};
    tt::tt_metal::detail::WriteToDeviceL1(device, dispatch_s_core, noc_xy_field_addr, noc_xy_data, CoreType::WORKER);

    std::vector<uint32_t> remote_state_data = {profiler_state_addr};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, dispatch_s_core, remote_state_field_addr, remote_state_data, CoreType::WORKER);

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device {}: wrote real-time profiler core info (noc_xy=0x{:x}, "
        "remote_state_addr=0x{:x}) "
        "to dispatch_s ({}, {})",
        device_id,
        profiler_noc_xy,
        profiler_state_addr,
        dispatch_s_core.x,
        dispatch_s_core.y);
}

void zero_profiler_l1(
    IDevice* device, CoreCoord profiler_core, uint32_t ring_buffer_addr, uint32_t msg_base_addr, uint32_t msg_size) {
    constexpr uint32_t kRingHeaderBytes = offsetof(RtProfilerRingBuffer, data);
    static_assert(kRingHeaderBytes % sizeof(uint32_t) == 0, "Ring header must be uint32-aligned");
    std::vector<uint32_t> zero_header(kRingHeaderBytes / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToDeviceL1(device, profiler_core, ring_buffer_addr, zero_header, CoreType::WORKER);

    std::vector<uint32_t> zero_msg(msg_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToDeviceL1(device, profiler_core, msg_base_addr, zero_msg, CoreType::WORKER);
}

struct ProfilerKernelAddrs {
    uint32_t ring_buffer = 0;
    uint32_t msg_base = 0;
    uint32_t config_field = 0;  // where the socket config address is published
    bool pcie_noc_defines = false;
    uint32_t pcie_noc_x = 0;
    uint32_t pcie_noc_y = 0;
};

// Wormhole's NCRISC pusher needs the PCIe core's NOC-0 coordinates baked in as defines; it translates them to NOC 1
// itself.
ProfilerKernelAddrs resolve_pcie_noc_defines(ContextId context_id, ChipId device_id, ProfilerKernelAddrs addrs) {
    const auto& metal = MetalContext::instance(context_id);
    if (metal.hal().get_arch() != tt::ARCH::WORMHOLE_B0) {
        return addrs;
    }
    const auto& cluster = metal.get_cluster();
    const auto& soc = cluster.get_soc_desc(cluster.get_associated_mmio_device(device_id));
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    TT_ASSERT(!pcie_cores.empty());
    addrs.pcie_noc_defines = true;
    addrs.pcie_noc_x = pcie_cores.front().x;
    addrs.pcie_noc_y = pcie_cores.front().y;
    return addrs;
}

// Publishes the socket config address only after launch, since the NCRISC pusher waits on it.
std::unique_ptr<Program> launch_profiler_kernels(
    IDevice* device,
    const Hal& hal,
    dispatch_core_manager& core_manager,
    CoreCoord profiler_core,
    const ProfilerKernelAddrs& addrs,
    uint32_t socket_config_addr,
    const std::string& reader_kernel_path,
    const std::string& push_kernel_path) {
    using ProfilerMsg = realtime_profiler_msgs::realtime_profiler_msg_t;
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    const ChipId device_id = device->id();

    uint32_t dispatch_noc_x = 0;
    uint32_t dispatch_noc_y = 0;
    uint32_t dispatch_data_a = 0;
    uint32_t dispatch_data_b = 0;
    if (core_manager.is_dispatcher_s_core_allocated(device_id, 0, 0)) {
        const tt_cxy_pair& dispatch_s_cxy = core_manager.dispatcher_s_core(device_id, 0, 0);
        const CoreCoord dispatch_s_virtual =
            device->virtual_core_from_logical_core(CoreCoord(dispatch_s_cxy.x, dispatch_s_cxy.y), CoreType::WORKER);
        dispatch_noc_x = dispatch_s_virtual.x;
        dispatch_noc_y = dispatch_s_virtual.y;
        dispatch_data_a =
            addrs.msg_base + static_cast<uint32_t>(factory.offset_of<ProfilerMsg>(ProfilerMsg::Field::kernel_start_a));
        dispatch_data_b =
            addrs.msg_base + static_cast<uint32_t>(factory.offset_of<ProfilerMsg>(ProfilerMsg::Field::kernel_start_b));
    }

    auto program = std::make_unique<Program>();

    DataMovementConfig brisc_config;
    brisc_config.processor = DataMovementProcessor::RISCV_0;
    brisc_config.noc = NOC::RISCV_0_default;
    brisc_config.defines["DISPATCH_CORE_NOC_X"] = std::to_string(dispatch_noc_x);
    brisc_config.defines["DISPATCH_CORE_NOC_Y"] = std::to_string(dispatch_noc_y);
    brisc_config.defines["DISPATCH_DATA_ADDR_A"] = std::to_string(dispatch_data_a);
    brisc_config.defines["DISPATCH_DATA_ADDR_B"] = std::to_string(dispatch_data_b);
    brisc_config.defines["RING_BUFFER_ADDR"] = std::to_string(addrs.ring_buffer);
    brisc_config.defines["REALTIME_PROFILER_MSG_ADDR"] = std::to_string(addrs.msg_base);
    CreateKernel(*program, reader_kernel_path, profiler_core, brisc_config);

    DataMovementConfig ncrisc_config;
    ncrisc_config.processor = DataMovementProcessor::RISCV_1;
    ncrisc_config.noc = NOC::RISCV_1_default;
    ncrisc_config.defines["RING_BUFFER_ADDR"] = std::to_string(addrs.ring_buffer);
    ncrisc_config.defines["REALTIME_PROFILER_MSG_ADDR"] = std::to_string(addrs.msg_base);
    if (addrs.pcie_noc_defines) {
        ncrisc_config.defines["RT_PROFILER_PCIE_NOC_X"] = std::to_string(addrs.pcie_noc_x);
        ncrisc_config.defines["RT_PROFILER_PCIE_NOC_Y"] = std::to_string(addrs.pcie_noc_y);
    }
    CreateKernel(*program, push_kernel_path, profiler_core, ncrisc_config);

    tt::tt_metal::detail::CompileProgram(device, *program, /*force_slow_dispatch=*/true);
    ::tt::tt_metal::detail::WriteRuntimeArgsToDevice(device, *program, /*force_slow_dispatch=*/true);
    ::tt::tt_metal::detail::LaunchProgram(
        device, *program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

    // realtime_profiler_msg_t is outside mailboxes_t, so LaunchProgram's writes do not race with config_buffer_addr;
    // ordering this write after the launch is intentional.
    std::vector<uint32_t> addr_data = {socket_config_addr};
    tt::tt_metal::detail::WriteToDeviceL1(device, profiler_core, addrs.config_field, addr_data, CoreType::WORKER);

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Device {}: launched real-time profiler BRISC+NCRISC kernels on core ({}, {}), "
        "ring_buffer_addr=0x{:x}, config_buffer_addr=0x{:x}",
        device_id,
        profiler_core.x,
        profiler_core.y,
        addrs.ring_buffer,
        socket_config_addr);
    return program;
}

// Evaluates against the device's owning context_id (not bare instance()) so a mock device isn't falsely enabled via
// the silicon DEFAULT_CONTEXT_ID fallback (#38445/#39849).
std::optional<CoreCoord> evaluate_realtime_profiler_eligibility(IDevice* device, ContextId context_id) {
    const auto device_id = device->id();
    auto& metal = MetalContext::instance(context_id);
    const auto& hal = metal.hal();
    const auto& cluster = metal.get_cluster();
    auto& dispatch_core_manager = metal.get_dispatch_core_manager();

    // Every refusal says what to change to re-enable; refuse_loudly where the configuration looks accidental rather
    // than chosen.
    const auto refuse = [&](std::string_view reason) {
        log_debug(tt::LogMetal, "Real-time profiler disabled on device {}: {}", device_id, reason);
        return std::optional<CoreCoord>{};
    };
    const auto refuse_loudly = [&](std::string_view reason) {
        log_warning(tt::LogMetal, "Real-time profiler disabled on device {}: {}", device_id, reason);
        return std::optional<CoreCoord>{};
    };

    // D2HSocket::init_host_buffer_hugepage dereferences a real PCIe hugepage.
    if (cluster.is_mock_or_emulated()) {
        return refuse("target is mock or emulated, and a D2H socket needs a real PCIe hugepage.");
    }
    // ttsim serves a clock read far slower than silicon: bring-up burns ~30s/chip with no usable calibration.
    if (cluster.get_target_device_type() == tt::TargetDevice::Simulator) {
        return refuse("target is the simulator, whose emulated PCIe cannot serve clock reads at a usable rate.");
    }
    if (!device->is_mmio_capable()) {
        return refuse("device is remote, and a D2H socket needs its sender core on a PCIe-connected chip.");
    }
    if (hal.get_supports_64_bit_pcie_addressing() && !cluster.is_iommu_enabled()) {
        return refuse(
            "this architecture addresses the D2H socket with 64-bit PCIe, which needs host IOMMU, and there is no "
            "hugepage fallback. Enable IOMMU to re-enable the profiler.");
    }
    if (metal.get_fabric_tensix_config() != tt_fabric::FabricTensixConfig::DISABLED) {
        return refuse(fmt::format(
            "the fabric tensix datamover is enabled (FabricTensixConfig={}, FabricUDMMode={}), and fabric_mux_core() "
            "drains the remaining dispatch-pool cores at fabric-init time. Reserving one more for the profiler tips "
            "small-pool chips into exhaustion. Disable the fabric tensix datamover to re-enable the profiler.",
            enchantum::to_string(metal.get_fabric_tensix_config()),
            enchantum::to_string(metal.get_fabric_udm_mode())));
    }
    const std::optional<tt_cxy_pair> reserved = dispatch_core_manager.get_reserved_realtime_profiler_core(device_id);
    if (!reserved.has_value()) {
        return refuse(
            "no tensix could be reserved. Dispatch is configured for ETH cores, which cannot run the profiler's BRISC "
            "kernel; switch to DispatchCoreConfig(DispatchCoreType::WORKER) to re-enable the profiler.");
    }
    if (metal.rtoptions().get_kernels_nullified()) {
        return refuse(
            "null-kernels mode is active, so the profiler kernel would be a stub that cannot answer host syncs, and "
            "there are no real kernels to profile.");
    }

    const CoreCoord core(reserved->x, reserved->y);
    const CoreCoord tensix_grid = cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    if (core.x >= tensix_grid.x || core.y >= tensix_grid.y) {
        return refuse_loudly(fmt::format(
            "reserved core ({}, {}) is outside the TENSIX logical grid ({}, {}).",
            core.x,
            core.y,
            tensix_grid.x,
            tensix_grid.y));
    }
    const uint32_t needed = tt::align(RealtimeProfilerRuntimeSizes::core_l1_size, hal.get_alignment(HalMemType::L1));
    const DeviceAddr l1_bank_size = device->allocator()->get_bank_size(BufferType::L1);
    if (l1_bank_size < needed) {
        return refuse_loudly(fmt::format(
            "the reserved core ({}, {}) has {} B of user-allocatable L1 against the {} B the profiler's L1 layout "
            "needs. Raise worker_l1_size by at least {} B, or leave it at the default, to re-enable the profiler.",
            core.x,
            core.y,
            l1_bank_size,
            needed,
            needed - l1_bank_size));
    }

    return core;
}

}  // namespace

RealtimeProfilerDevice::RealtimeProfilerDevice() = default;
RealtimeProfilerDevice::~RealtimeProfilerDevice() = default;
RealtimeProfilerDevice::RealtimeProfilerDevice(RealtimeProfilerDevice&&) noexcept = default;

std::vector<RealtimeProfilerDevice> initialize_realtime_profiler_devices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id) {
    std::vector<RealtimeProfilerDevice> devices;
    const auto& hal = MetalContext::instance(context_id).hal();
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    using ProfilerMsg = realtime_profiler_msgs::realtime_profiler_msg_t;
    const auto msg_field_addr = [&factory](uint32_t base, ProfilerMsg::Field field) {
        return base + static_cast<uint32_t>(factory.offset_of<ProfilerMsg>(field));
    };
    const auto& dispatch_mem_map = MetalContext::instance(context_id).dispatch_mem_map();
    // TODO: When realtime profiler is supported on Quasar, we'll need to pass in the command queue id(s).
    const uint32_t realtime_profiler_base_addr =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG, /*cq_id=*/0);
    // RealtimeProfilerCoreL1 (ring + D2H sender config) sits past the dispatch carve-outs; the core is off the L1 bank
    // table so the allocator never lands here.
    const uint32_t rt_profiler_core_l1_base =
        dispatch_mem_map.get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, /*cq_id=*/0);
    const auto rt_profiler_core_l1_addrs = compute_rt_profiler_core_l1_addrs(rt_profiler_core_l1_base);

    TT_FATAL(
        RT_PROFILER_SOCKET_CONFIG_SIZE >= distributed::D2HSocket::required_config_buffer_size(),
        "RT_PROFILER_SOCKET_CONFIG_SIZE ({} B) is smaller than distributed::D2HSocket's required config "
        "buffer size ({} B). Bump RT_PROFILER_SOCKET_CONFIG_SIZE in "
        "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp and rebuild.",
        RT_PROFILER_SOCKET_CONFIG_SIZE,
        distributed::D2HSocket::required_config_buffer_size());
    const uint32_t profiler_msg_config_field_addr =
        msg_field_addr(realtime_profiler_base_addr, ProfilerMsg::Field::config_buffer_addr);

    auto& dispatch_core_manager = MetalContext::instance(context_id).get_dispatch_core_manager();
    const std::string realtime_profiler_kernel_path = "tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp";
    const std::string realtime_profiler_push_kernel_path =
        "tt_metal/impl/dispatch/kernels/cq_realtime_profiler_push.cpp";

    // Empty for a device that cannot run the profiler, or whose socket could not be created.
    const auto initialize_device = [&](const distributed::MeshCoordinate& coord,
                                       IDevice* device) -> std::optional<RealtimeProfilerDevice> {
        const auto device_id = device->id();
        const std::optional<CoreCoord> profiler_core = evaluate_realtime_profiler_eligibility(device, context_id);
        if (!profiler_core.has_value()) {
            return std::nullopt;
        }
        const CoreCoord realtime_profiler_core = *profiler_core;

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Using reserved tensix ({}, {}) for real-time profiler on device {}",
            realtime_profiler_core.x,
            realtime_profiler_core.y,
            device_id);

        RealtimeProfilerDevice dev_state;
        dev_state.device = device;
        dev_state.chip_id = device_id;
        dev_state.realtime_profiler_core = realtime_profiler_core;
        dev_state.core_l1 = rt_profiler_core_l1_addrs;

        log_debug(
            tt::LogMetal,
            "[Real-time profiler] Initializing real-time profiler D2H socket for device {} on distributed::MeshDevice "
            "{}",
            device_id,
            mesh_device->id());

        dev_state.socket = create_d2h_socket(
            mesh_device,
            distributed::MeshCoreCoord{coord, realtime_profiler_core},
            dev_state.core_l1.socket_config,
            device_id);
        if (!dev_state.socket) {
            return std::nullopt;
        }

        // Before the kernels launch, so nothing this discards can be a real record: undefined bytes on a fresh
        // MeshDevice, or pages an SHM-recovered FIFO carried over, would otherwise be decoded as bogus records.
        const uint32_t stale_pages = dev_state.socket->discard_pending_pages();
        if (stale_pages > 0) {
            log_debug(tt::LogMetal, "[Real-time profiler] Device {} discarded {} stale pages", device_id, stale_pages);
        }

        publish_profiler_core_to_dispatch_s(
            device,
            hal,
            dispatch_core_manager,
            realtime_profiler_core,
            msg_field_addr(realtime_profiler_base_addr, ProfilerMsg::Field::realtime_profiler_core_noc_xy),
            msg_field_addr(realtime_profiler_base_addr, ProfilerMsg::Field::realtime_profiler_remote_state_addr),
            msg_field_addr(realtime_profiler_base_addr, ProfilerMsg::Field::realtime_profiler_state));

        zero_profiler_l1(
            device,
            realtime_profiler_core,
            dev_state.core_l1.ring_buffer,
            realtime_profiler_base_addr,
            factory.size_of<realtime_profiler_msgs::realtime_profiler_msg_t>());

        ProfilerKernelAddrs kernel_addrs;
        kernel_addrs.ring_buffer = dev_state.core_l1.ring_buffer;
        kernel_addrs.msg_base = realtime_profiler_base_addr;
        kernel_addrs.config_field = profiler_msg_config_field_addr;
        dev_state.realtime_profiler_program = launch_profiler_kernels(
            device,
            hal,
            dispatch_core_manager,
            realtime_profiler_core,
            resolve_pcie_noc_defines(context_id, device_id, kernel_addrs),
            dev_state.socket->get_config_buffer_address(),
            realtime_profiler_kernel_path,
            realtime_profiler_push_kernel_path);

        return dev_state;
    };

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        IDevice* device = mesh_device->get_device(coord);
        std::optional<RealtimeProfilerDevice> dev_state = initialize_device(coord, device);
        // Unconditional: whoever waits on this flag hangs if bring-up bailed out without setting it.
        MetalContext::instance(context_id).device_manager()->mark_rt_profiler_device_init_complete(device->id());
        if (dev_state.has_value()) {
            devices.push_back(std::move(*dev_state));
        }
    }
    return devices;
}

}  // namespace tt::tt_metal
