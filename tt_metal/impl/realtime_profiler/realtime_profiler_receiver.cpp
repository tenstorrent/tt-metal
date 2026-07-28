// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
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
#include <llrt/tt_cluster.hpp>

#include "context/metal_context.hpp"
#include "device/device_manager.hpp"
#include "dispatch/command_queue_common.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "dispatch/dispatch_mem_map.hpp"
#include "dispatch/system_memory_manager.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include "llrt/hal.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/dispatch/data_collector.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

namespace tt::tt_metal {

namespace {

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

constexpr size_t kMaxConsumerBatchPerDevice =
    1u << 15;                                      // records one callback may be handed at a time, per attached device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;  // hard ceiling on the above
constexpr size_t kRingHeadroomBatches = 4;         // batches of backlog the ring absorbs while a consumer works
constexpr size_t kMaxRingCapacity = 1u << 22;      // hard ceiling on the ring size

// How often each device is resynced. Devices take their turn one per tick rather than all together, so the interval
// sets the per-device cadence while the tick sets how long draining can be blocked by a resync.
constexpr auto kClockSyncInterval = std::chrono::milliseconds(50);

inline RealtimeProfilerCoreL1Addrs compute_rt_profiler_core_l1_addrs(uint32_t base) {
    return {
        .ring_buffer = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, ring)),
        .socket_config = base + static_cast<uint32_t>(offsetof(RealtimeProfilerCoreL1, socket_config)),
    };
}

// Constructs the device's D2H socket.
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
    uint32_t dispatch_data_a = 0;
    uint32_t dispatch_data_b = 0;
    uint32_t dispatch_noc_x = 0;
    uint32_t dispatch_noc_y = 0;
    // Wormhole needs the PCIe core's NOC-0 coordinates baked in; the NCRISC kernel translates them to NOC 1.
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

// Builds and launches the BRISC reader + NCRISC pusher, then publishes the socket config address the NCRISC waits on.
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

// Consolidated eligibility check; logs the disable reason. Evaluates against the device's owning context_id (not bare
// instance()) so a mock device isn't falsely enabled via the silicon DEFAULT_CONTEXT_ID fallback (#38445/#39849).
// Checks: not mock/emulated, MMIO-capable, IOMMU if 64-bit PCIe, fabric tensix datamover off, a tensix reserved and
// in-grid, kernels not nullified, L1 bank fits the layout.
std::optional<CoreCoord> evaluate_realtime_profiler_eligibility(IDevice* device, ContextId context_id) {
    auto device_id = device->id();
    auto& metal = MetalContext::instance(context_id);
    const auto& hal = metal.hal();
    const auto& cluster = metal.get_cluster();
    auto& dispatch_core_manager = metal.get_dispatch_core_manager();

    // Gate mock/emulated targets: distributed::D2HSocket::init_host_buffer_hugepage dereferences a real PCIe hugepage
    // absent there.
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

    return core;
}

// Runs an action when the enclosing scope ends, however it ends.
template <typename Fn>
class OnScopeExit {
public:
    explicit OnScopeExit(Fn fn) : fn_(std::move(fn)) {}
    ~OnScopeExit() { fn_(); }
    OnScopeExit(const OnScopeExit&) = delete;
    OnScopeExit& operator=(const OnScopeExit&) = delete;

private:
    Fn fn_;
};

template <typename Fn>
void calibrate_devices_concurrently(const std::vector<size_t>& indices, Fn&& fn) {
    std::vector<std::thread> workers;
    workers.reserve(indices.size());
    for (size_t di : indices) {
        workers.emplace_back([di, &fn] {
            try {
                fn(di);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal, "[Real-time profiler] Device init sync failed, skipping device: {}", e.what());
            } catch (...) {
                log_warning(tt::LogMetal, "[Real-time profiler] Device init sync failed, skipping device");
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

}  // namespace

RealtimeProfilerReceiver::DeviceState::DeviceState() = default;
RealtimeProfilerReceiver::DeviceState::~DeviceState() = default;
RealtimeProfilerReceiver::DeviceState::DeviceState(DeviceState&&) noexcept = default;

std::vector<uint64_t> RealtimeProfilerReceiver::num_resyncs_per_device() const {
    std::vector<uint64_t> counts;
    counts.reserve(num_resyncs_.size());
    for (const auto& count : num_resyncs_) {
        counts.push_back(count.load(std::memory_order_relaxed));
    }
    return counts;
}

uint32_t RealtimeProfilerReceiver::host_fifo_capacity_pages() const { return RealtimeProfilerRuntimeSizes::fifo_pages; }

uint32_t RealtimeProfilerReceiver::read_ring_full_wait_count() {
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

void RealtimeProfilerReceiver::publish_pages(
    const DeviceState& dev_state,
    std::chrono::steady_clock::time_point now,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double frequency = dev_state.clock_sync.frequency();
    const experimental::ProgramRealtimeClockSync clock_sync = dev_state.clock_sync.mapping(now);
    const DataCollector* const data_collector = data_collector_;
    uint64_t malformed = 0;
    for (uint32_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = page_buf + page * kPageWords;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++malformed;
            continue;
        }
        records.push_back(ProgramRealtimeRecord{
            .runtime_id = rp[2],
            .chip_id = chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = frequency,
            .clock_sync = clock_sync,
            .kernel_sources = data_collector->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])),
        });
    }
    if (malformed != 0) {
        num_malformed_records_.fetch_add(malformed, std::memory_order_relaxed);
        if (const auto dropped = malformed_record_warns_.record(now, malformed)) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} dropped {} record(s) whose end timestamp preceded their start; these "
                "were not delivered to consumers ({} in total)",
                dev_state.chip_id,
                *dropped,
                num_malformed_records_.load(std::memory_order_relaxed));
        }
    }
    if (records.empty()) {
        return;
    }
    num_published_records_.fetch_add(records.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_.writer().publish_batch(records);
}

void RealtimeProfilerReceiver::resync_next_device(std::chrono::steady_clock::time_point now) {
    const size_t index = next_resync_device_;
    DeviceState& dev_state = devices_[index];
    next_resync_device_ = (index + 1) % devices_.size();
    num_resyncs_[index].fetch_add(1, std::memory_order_relaxed);
    if (!dev_state.clock_sync.resync(now)) {
        if (const auto total = probe_timeout_warns_.record(now, 1)) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] {} clock resync probe(s) went unanswered; keeping the previous mapping on the "
                "affected devices",
                *total);
        }
    }
}

namespace {

// Records one callback may be handed, and the backlog the ring absorbs behind it. Both scale with the number of
// devices feeding the ring.
size_t consumer_batch_records_for(size_t num_devices) {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * num_devices);
}

}  // namespace

std::unique_ptr<RealtimeProfilerReceiver> RealtimeProfilerReceiver::create(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const ContextId context_id = mesh_device->impl().get_context_id();
    auto devices =
        initialize_devices(mesh_device, context_id, d2h_uses_hugepage_fallback(MetalContext::instance(context_id)));
    if (devices.empty()) {
        log_debug(
            tt::LogMetal, "[Real-time profiler] No local devices found in mesh, skipping real-time profiler setup");
        return nullptr;
    }
    register_builtin_realtime_profiler_consumers();
    return std::unique_ptr<RealtimeProfilerReceiver>(new RealtimeProfilerReceiver(context_id, std::move(devices)));
}

RealtimeProfilerReceiver::RealtimeProfilerReceiver(ContextId context_id, std::vector<DeviceState> devices) :
    context_id_(context_id),
    d2h_hugepage_fallback_(d2h_uses_hugepage_fallback(MetalContext::instance(context_id))),
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    realtime_profiler_service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    num_resyncs_(devices_.size()),
    ring_(std::min(kMaxRingCapacity, consumer_batch_records_for(devices_.size()) * kRingHeadroomBatches)) {
    run_init_sync();
    realtime_profiler_service_->attach_ring(ring_, consumer_batch_records_for(devices_.size()));

    try {
        receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run, this);
    } catch (...) {
        realtime_profiler_service_->detach_ring(ring_);
        throw;
    }
}

std::vector<RealtimeProfilerReceiver::DeviceState> RealtimeProfilerReceiver::initialize_devices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id, bool hugepage_fallback) {
    std::vector<DeviceState> devices;
    // HAL offsets are the same for all devices (same arch).
    const auto& hal = MetalContext::instance(context_id).hal();
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    // TODO: When realtime profiler is supported on Quasar, we'll need to pass in the command queue id(s) here.
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

    // RT_PROFILER_SOCKET_CONFIG_SIZE has headroom over today's SocketSenderSize, but assert
    // it here so a future growth of the sender config triggers a deterministic startup failure.
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

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }

        IDevice* device = mesh_device->get_device(coord);
        auto device_id = device->id();

        // Whoever waits on this flag hangs if a bailout below does not set it.
        OnScopeExit mark_init_complete([context_id, device_id] {
            MetalContext::instance(context_id).device_manager()->mark_rt_profiler_device_init_complete(device_id);
        });

        const std::optional<CoreCoord> profiler_core = evaluate_realtime_profiler_eligibility(device, context_id);
        if (!profiler_core.has_value()) {
            continue;
        }
        const CoreCoord realtime_profiler_core = *profiler_core;

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
            continue;
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

        dev_state.clock_sync.configure(RealtimeProfilerClockSyncConfig{
            .context_id = context_id,
            .device = device,
            .mesh_device = mesh_device.get(),
            .profiler_core = dev_state.realtime_profiler_core,
            .mesh_coord = dev_state.mesh_coord,
            .hugepage_fallback = hugepage_fallback,
            .msg_base_addr = realtime_profiler_base_addr,
        });

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

        devices.push_back(std::move(dev_state));
    }
    return devices;
}

void RealtimeProfilerReceiver::run_init_sync() {
    constexpr uint32_t kInitSyncMaxRetries = 3;
    constexpr auto kInitSyncRetryDelay = std::chrono::milliseconds(500);
    std::vector<size_t> device_indices(devices_.size());
    std::iota(device_indices.begin(), device_indices.end(), 0);

    calibrate_devices_concurrently(device_indices, [&](size_t di) {
        auto& dev_state = devices_[di];
        if (dev_state.clock_sync.try_restore_calibration(std::chrono::steady_clock::now())) {
            return;
        }
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
            // Undefined bytes on a fresh MeshDevice (or SHM-recovered stale pages) would otherwise be decoded by the
            // receiver as bogus records.
            const uint32_t stale_pages = dev_state.socket->discard_pending_pages();
            if (stale_pages > 0) {
                log_debug(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} discarded {} stale pages before sync",
                    dev_state.chip_id,
                    stale_pages);
            }
            if (dev_state.clock_sync.run_fit()) {
                break;
            }
        }
    });
}

uint32_t RealtimeProfilerReceiver::drain_device_pages(
    DeviceState& dev_state,
    std::chrono::steady_clock::time_point now,
    std::vector<uint32_t>& page_buf,
    std::vector<ProgramRealtimeRecord>& record_buf) {
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
        return 0;
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    dev_state.socket->read(page_buf.data(), num_pages_to_read);
    publish_pages(dev_state, now, page_buf.data(), num_pages_to_read, record_buf);
    return num_pages_to_read;
}

uint64_t RealtimeProfilerReceiver::run_loop(
    std::vector<uint32_t>& page_buf, std::vector<ProgramRealtimeRecord>& record_buf) {
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;
    const auto clock_sync_tick = kClockSyncInterval / devices_.size();
    auto last_clock_sync = std::chrono::steady_clock::now();
#if defined(TRACY_ENABLE) && TT_TRACY_CATEGORY_RT_PROFILER
    constexpr auto kFifoPlotInterval = std::chrono::milliseconds(10);
    auto last_fifo_plot = std::chrono::steady_clock::now();
#endif
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const uint32_t num_pages = drain_all_devices(now, page_buf, record_buf);
        num_pages_received += num_pages;
        if (now - last_clock_sync >= clock_sync_tick) {
            resync_next_device(now);
            last_clock_sync = now;
        }
#if defined(TRACY_ENABLE) && TT_TRACY_CATEGORY_RT_PROFILER
        if (now - last_fifo_plot >= kFifoPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(fifo_pages_window_max_));
            fifo_pages_window_max_ = 0;
            int64_t worst_sync_error_ns = 0;
            for (const auto& dev_state : devices_) {
                worst_sync_error_ns = std::max(
                    worst_sync_error_ns, static_cast<int64_t>(dev_state.clock_sync.mapping(now).sync_error_ns));
            }
            TTTracyPlotD(RT_PROFILER, "RT sync error (us)", static_cast<double>(worst_sync_error_ns) / 1000.0);
            last_fifo_plot = now;
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

uint32_t RealtimeProfilerReceiver::drain_all_devices(
    std::chrono::steady_clock::time_point now,
    std::vector<uint32_t>& page_buf,
    std::vector<ProgramRealtimeRecord>& record_buf) {
    uint32_t num_pages = 0;
    for (auto& dev_state : devices_) {
        try {
            num_pages += drain_device_pages(dev_state, now, page_buf, record_buf);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal, "[Real-time profiler] Exception draining device {}: {}", dev_state.chip_id, e.what());
        }
    }
    if (num_pages > 0) {
        realtime_profiler_service_->wake_consumers();
    }
    return num_pages;
}

uint64_t RealtimeProfilerReceiver::drain_on_shutdown(
    std::vector<uint32_t>& page_buf, std::vector<ProgramRealtimeRecord>& record_buf) {
    constexpr uint32_t kShutdownDrainQuietRounds = 10;
    constexpr auto kShutdownDrainQuietBackoff = std::chrono::milliseconds(1);
    uint64_t num_pages_drained = 0;
    uint32_t quiet_rounds = 0;
    while (quiet_rounds < kShutdownDrainQuietRounds) {
        const uint32_t num_pages = drain_all_devices(std::chrono::steady_clock::now(), page_buf, record_buf);
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

void RealtimeProfilerReceiver::run() {
    tracy::SetThreadName("RealtimeProfiler");
#if defined(__linux__)
    ::prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
    log_debug(tt::LogMetal, "[Real-time profiler] Receiver thread started for {} devices", devices_.size());

    // One buffer pair for the thread's whole life; the steady-state loop and the shutdown drain run in sequence.
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * kPageWords);
    std::vector<ProgramRealtimeRecord> record_buf;
    record_buf.reserve(kMaxSocketPagesPerRead);

    const uint64_t num_pages_received = run_loop(page_buf, record_buf);

    // Signal the device kernels to terminate only after the drain loop above has exited, so no handshake fires against
    // a core that is already tearing down (that would just time out). The push kernel then delivers its last pages,
    // which the shutdown drain below collects once traffic goes quiet.
    for (auto& dev_state : devices_) {
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

    const uint64_t num_pages_drained = drain_on_shutdown(page_buf, record_buf);

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown)",
        num_pages_received + num_pages_drained,
        num_pages_drained);
}

RealtimeProfilerReceiver::~RealtimeProfilerReceiver() { shutdown(); }

void RealtimeProfilerReceiver::shutdown() {
    if (receiver_thread_.joinable()) {
        stop_.store(true, std::memory_order_release);
        receiver_thread_.join();
    }

    // detach_ring is idempotent, so a second shutdown() is harmless.
    realtime_profiler_service_->detach_ring(ring_);

    for (const auto& dev_state : devices_) {
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

}  // namespace tt::tt_metal
