// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
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
#include "tt_metal/common/env_lib.hpp"
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
    static constexpr uint32_t page_words = page_size / sizeof(uint32_t);
    static constexpr uint32_t fifo_size = fifo_pages * page_size;  // pinned-host FIFO, in bytes (2 MiB)
    static constexpr uint32_t core_l1_size = sizeof(RealtimeProfilerCoreL1);
};

static_assert(
    RealtimeProfilerRuntimeSizes::fifo_pages >= RT_PROFILER_RING_CAPACITY,
    "Host D2H FIFO must be at least as deep as the device ring (RT_PROFILER_RING_CAPACITY)");

constexpr uint32_t kMaxSocketPagesPerRead = 1024;

// The probe history must out-span anything still in flight: a device is probed once per drained batch and, when its
// FIFO is empty, once per sync interval, so a record that has ended sees at most pipeline-depth / batch-size probes
// (plus a handful racing the push) before it is decoded. This is what lets a record's end always find the pair of
// probes around it, however far its start may reach back -- whatever horizon the ring was sized for.
static_assert(
    RealtimeProfilerClockMapping::kProbeHistoryCapacity >=
        8 * (RealtimeProfilerRuntimeSizes::fifo_pages + RT_PROFILER_RING_CAPACITY) / kMaxSocketPagesPerRead,
    "The probe history could lap past an undecoded record's end");

// Floor on how often a repeating fault is logged.
constexpr auto kWarnInterval = std::chrono::seconds(30);

// How often the reporter thread turns pending telemetry into log lines.
constexpr auto kReporterInterval = std::chrono::seconds(1);

constexpr auto kSyncCostReportInterval = std::chrono::seconds(5);

constexpr auto kDrainGapReportThreshold = std::chrono::milliseconds(5);

constexpr size_t kMaxConsumerBatchPerDevice =
    1u << 15;                                      // records one callback may be handed at a time, per attached device
constexpr size_t kMaxConsumerBatchCap = 1u << 20;  // hard ceiling on the above
constexpr size_t kRingHeadroomBatches = 4;         // batches of backlog the ring absorbs while a consumer works
constexpr size_t kMaxRingCapacity = 1u << 22;      // hard ceiling on the ring size

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

    // The profiler turning itself off is invisible otherwise, so every refusal says what to change to re-enable it.
    // Warning rather than debug where the configuration looks accidental rather than chosen.
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

size_t consumer_batch_records_for(size_t num_devices) {
    return std::min(kMaxConsumerBatchCap, kMaxConsumerBatchPerDevice * num_devices);
}

}  // namespace

RealtimeProfilerReceiver::DeviceState::DeviceState() = default;
RealtimeProfilerReceiver::DeviceState::~DeviceState() = default;
RealtimeProfilerReceiver::DeviceState::DeviceState(DeviceState&&) noexcept = default;

void RealtimeProfilerReceiver::run_reporter() {
    tracy::SetThreadName("RtProfReporter");

    auto sum_costs = [this] {
        RealtimeProfilerClockSync::Cost total;
        for (const auto& dev_state : devices_) {
            total += dev_state.clock_sync->cost();
        }
        return total;
    };
    auto last_cost_report = std::chrono::steady_clock::now();
    RealtimeProfilerClockSync::Cost cost_at_last_report = sum_costs();
    std::vector<uint64_t> malformed_seen(devices_.size(), 0);
    std::vector<uint64_t> exceptions_seen(devices_.size(), 0);
    std::vector<bool> capacity_warned(devices_.size(), false);
    std::chrono::steady_clock::time_point last_gap_warn{};
    std::chrono::steady_clock::time_point last_malformed_warn{};
    std::chrono::steady_clock::time_point last_exception_warn{};

    const auto emit_pending = [&](std::chrono::steady_clock::time_point now) {
        // While throttled the pack is left in place, still taking maxes, so the eventual line carries the worst gap.
        if (worst_drain_gap_.load(std::memory_order_relaxed) != 0 && now - last_gap_warn >= kWarnInterval) {
            const uint64_t pack = worst_drain_gap_.exchange(0, std::memory_order_relaxed);
            last_gap_warn = now;
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Receiver drain stalled {} us between passes, {} us of it inside clock reads; "
                "the FIFO has peaked at {} of {} pages this run",
                pack >> 32,
                pack & 0xFFFFFFFFu,
                peak_fifo_pages_.load(std::memory_order_relaxed),
                RealtimeProfilerRuntimeSizes::fifo_pages);
        }

        for (size_t i = 0; i < devices_.size(); ++i) {
            const DeviceState& dev_state = devices_[i];
            DeviceTelemetry& telemetry = *dev_state.telemetry;
            if (!capacity_warned[i] && telemetry.fifo_reached_capacity.load(std::memory_order_relaxed)) {
                capacity_warned[i] = true;
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} D2H FIFO reached capacity ({} pages); profiler data may be "
                    "dropped",
                    dev_state.chip_id,
                    RealtimeProfilerRuntimeSizes::fifo_pages);
            }
            const uint64_t malformed = telemetry.malformed_records.load(std::memory_order_relaxed);
            if (malformed != malformed_seen[i] && now - last_malformed_warn >= kWarnInterval) {
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {} dropped {} corrupt record(s) -- an end timestamp preceding its "
                    "start, or timestamps predating every retained clock probe; these were not delivered to consumers "
                    "({} in total)",
                    dev_state.chip_id,
                    malformed - malformed_seen[i],
                    malformed);
                malformed_seen[i] = malformed;
                last_malformed_warn = now;
            }
            const uint64_t exceptions = telemetry.drain_exceptions.load(std::memory_order_relaxed);
            if (exceptions != exceptions_seen[i] && now - last_exception_warn >= kWarnInterval) {
                std::string latest;
                {
                    const std::lock_guard lock(telemetry.last_drain_error_mutex);
                    latest = telemetry.last_drain_error;
                }
                log_warning(
                    tt::LogMetal,
                    "[Real-time profiler] Device {}: {} exception(s) while draining since the last report; latest: {}",
                    dev_state.chip_id,
                    exceptions - exceptions_seen[i],
                    latest);
                exceptions_seen[i] = exceptions;
                last_exception_warn = now;
            }
        }

        // `busy` is time inside resync(), which is time the drain thread was not draining, so the fraction is the
        // share of the drain loop the sync path takes -- the number to watch when changing probe cadence or count.
        if (now - last_cost_report >= kSyncCostReportInterval) {
            const RealtimeProfilerClockSync::Cost total = sum_costs();
            const RealtimeProfilerClockSync::Cost cost = total.since(cost_at_last_report);
            const auto window = now - last_cost_report;
            cost_at_last_report = total;
            last_cost_report = now;
            if (cost.resyncs != 0) {
                log_info(
                    tt::LogMetal,
                    "[Real-time profiler] Sync cost over {}s across {} device(s): {} resyncs, {} clock reads ({:.4f} "
                    "per resync), {:.2f}us mean per resync, {:.2f}% of the receiver thread",
                    std::chrono::duration<double>{window}.count(),
                    devices_.size(),
                    cost.resyncs,
                    cost.clock_reads,
                    static_cast<double>(cost.clock_reads) / static_cast<double>(cost.resyncs),
                    std::chrono::duration<double, std::micro>{cost.busy}.count() / static_cast<double>(cost.resyncs),
                    100.0 * std::chrono::duration<double>{cost.busy}.count() /
                        std::chrono::duration<double>{window}.count());
            }
        }
    };

    std::unique_lock lock(reporter_mutex_);
    while (!reporter_cv_.wait_for(lock, kReporterInterval, [this] { return stop_.load(std::memory_order_acquire); })) {
        emit_pending(std::chrono::steady_clock::now());
    }
    // One last sweep so telemetry from the final second still reaches the log.
    emit_pending(std::chrono::steady_clock::now());
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

// Decodes what a read returned and publishes it, each record stamped from the secant between the two probes that
// surround it. Nothing is held back: the probe taken after the read is past every record in it, so the pair a record
// needs already exists by the time it is decoded.
bool RealtimeProfilerReceiver::publish_pages(
    DeviceState& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch) {
    TTZoneScopedDN(RT_PROFILER, "PublishBatch");
    const size_t num_pages = pages.size() / RealtimeProfilerRuntimeSizes::page_words;
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    const DataCollector* const data_collector = data_collector_;
    uint64_t rejected = 0;
    batch.clear();

    for (size_t page = 0; page < num_pages; ++page) {
        const uint32_t* rp = pages.data() + page * RealtimeProfilerRuntimeSizes::page_words;
        const uint64_t start_timestamp = (static_cast<uint64_t>(rp[0]) << 32) | rp[1];
        const uint64_t end_timestamp = (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1];
        if (end_timestamp < start_timestamp) {
            ++rejected;
            continue;
        }
        // Only fails when both timestamps predate every retained probe, which a real record cannot; see map_record.
        const auto mapping = dev_state.clock_sync->map_record(start_timestamp, end_timestamp);
        if (!mapping.has_value()) {
            ++rejected;
            continue;
        }
        batch.push_back(ProgramRealtimeRecord{
            .runtime_id = rp[2],
            .chip_id = dev_state.chip_id,
            .start_timestamp = start_timestamp,
            .end_timestamp = end_timestamp,
            .frequency = mapping->frequency,
            .clock_sync = mapping->clock_sync,
            .kernel_sources = data_collector->GetKernelSourcesForRuntimeId(static_cast<uint16_t>(rp[2])),
        });
    }

    if (rejected != 0) {
        num_malformed_records_.fetch_add(rejected, std::memory_order_relaxed);
        dev_state.telemetry->malformed_records.fetch_add(rejected, std::memory_order_relaxed);
    }
    if (batch.empty()) {
        return false;
    }
    num_published_records_.fetch_add(batch.size(), std::memory_order_relaxed);
    num_published_batches_.fetch_add(1, std::memory_order_relaxed);
    ring_.writer().publish_batch(std::span<const ProgramRealtimeRecord>(batch));
    return true;
}

std::unique_ptr<RealtimeProfilerReceiver> RealtimeProfilerReceiver::create(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const ContextId context_id = mesh_device->impl().get_context_id();
    auto devices = initialize_devices(mesh_device, context_id);
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
    data_collector_(MetalContext::instance(context_id).data_collector().get()),
    realtime_profiler_service_(&realtime_profiler_service()),
    devices_(std::move(devices)),
    ring_(std::min(kMaxRingCapacity, consumer_batch_records_for(devices_.size()) * kRingHeadroomBatches)) {
    // Serial: a warm-up is a few probes a sync interval apart, so a whole mesh costs milliseconds. It replaced a
    // half-second per-device fit, which is what the concurrency here existed for.
    for (DeviceState& dev_state : devices_) {
        dev_state.clock_sync->warm_up();
    }
    // Resized before being cleared, not just reserved: the pages have to be touched here, because a first-touch fault
    // on the drain thread waits on mmap_lock the same way an allocation does.
    publish_batch_.resize(kMaxSocketPagesPerRead);
    publish_batch_.clear();
    realtime_profiler_service_->attach_producer(*this);

    try {
        reporter_thread_ = std::thread(&RealtimeProfilerReceiver::run_reporter, this);
        receiver_thread_ = std::thread(&RealtimeProfilerReceiver::run, this);
    } catch (...) {
        {
            const std::lock_guard lock(reporter_mutex_);
            stop_.store(true, std::memory_order_release);
        }
        reporter_cv_.notify_all();
        if (reporter_thread_.joinable()) {
            reporter_thread_.join();
        }
        realtime_profiler_service_->detach_producer(*this);
        throw;
    }
}

size_t RealtimeProfilerReceiver::max_batch_records() const { return consumer_batch_records_for(devices_.size()); }

RealtimeProfilerRecordRing::Reader RealtimeProfilerReceiver::make_reader() { return ring_.make_reader(); }

void RealtimeProfilerReceiver::wait_until_no_readers() { ring_.wait_until_no_readers(); }

std::vector<RealtimeProfilerReceiver::DeviceState> RealtimeProfilerReceiver::initialize_devices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id) {
    std::vector<DeviceState> devices;
    // HAL offsets are the same for all devices (same arch).
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

    // Empty for a device that cannot run the profiler, or whose socket could not be created.
    const auto initialize_device = [&](const distributed::MeshCoordinate& coord,
                                       IDevice* device) -> std::optional<DeviceState> {
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

        DeviceState dev_state;
        dev_state.device = device;
        dev_state.chip_id = device_id;
        dev_state.realtime_profiler_core = realtime_profiler_core;
        dev_state.core_l1 = rt_profiler_core_l1_addrs;
        dev_state.telemetry = std::make_unique<DeviceTelemetry>();

        // Constructed before anything is published to dispatch_s or launched, so a device whose clock register cannot
        // be mapped is skipped rather than left half configured. There is no slower read to fall back to; see
        // has_direct_clock_read.
        dev_state.clock_sync =
            std::make_unique<RealtimeProfilerClockSync>(context_id, device, dev_state.realtime_profiler_core);
        if (!dev_state.clock_sync->has_direct_clock_read()) {
            log_warning(
                tt::LogMetal,
                "Real-time profiler disabled on device {}: the profiler core's clock register could not be mapped into "
                "a UC TLB window, and every published host time is anchored on a read of it.",
                device_id);
            return std::nullopt;
        }

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
        std::optional<DeviceState> dev_state = initialize_device(coord, device);
        // Unconditional: whoever waits on this flag hangs if bring-up bailed out without setting it.
        MetalContext::instance(context_id).device_manager()->mark_rt_profiler_device_init_complete(device->id());
        if (dev_state.has_value()) {
            devices.push_back(std::move(*dev_state));
        }
    }
    return devices;
}

// Three high-water marks over the same samples, because each has one reader on its own cadence and each clears what
// it reads: all-time for "did this ever back up", one for the Tracy plot's 50ms samples, one for whatever periodic
// report is watching. Sharing an accumulator between two of them made a reader's number mean the other's window.
void RealtimeProfilerReceiver::note_fifo_depth(uint32_t available) {
    // Single writer, so a load-then-store is a max.
    if (available > peak_fifo_pages_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_.store(available, std::memory_order_relaxed);
    }
    if (available > peak_fifo_pages_since_report_.load(std::memory_order_relaxed)) {
        peak_fifo_pages_since_report_.store(available, std::memory_order_relaxed);
    }
    fifo_pages_window_max_ = std::max(fifo_pages_window_max_, available);
}

RealtimeProfilerReceiver::DrainResult RealtimeProfilerReceiver::drain_device_pages(
    DeviceState& dev_state, std::vector<uint32_t>& page_buf) {
    const uint32_t available = dev_state.socket->pages_available();
    note_fifo_depth(available);
    if (available >= RealtimeProfilerRuntimeSizes::fifo_pages) {
        dev_state.telemetry->fifo_reached_capacity.store(true, std::memory_order_relaxed);
    }
    if (available == 0) {
        return {};
    }
    const uint32_t num_pages_to_read = std::min(available, kMaxSocketPagesPerRead);
    {
        TTZoneScopedDN(RT_PROFILER, "SocketRead");
        TTZoneValueD(RT_PROFILER, num_pages_to_read);
        dev_state.socket->read(page_buf.data(), num_pages_to_read);
    }

    // The probe goes here, after the read and before anything is placed: every record this read returned was pushed to
    // the FIFO before the read, so it completed before the read, so this probe is past all of them, and the previous
    // one is before them. That is a bracketing pair for the whole batch obtained without waiting for anything, which is
    // the reason this path has no staging buffer, no publication gate and no deadline in it.
    {
        TTZoneScopedDN(RT_PROFILER, "ProbeAfterRead");
        TTZoneValueD(RT_PROFILER, dev_state.chip_id);
        pass_sync_busy_ += dev_state.clock_sync->resync();
    }

    const bool published = publish_pages(
        dev_state,
        std::span(page_buf).first(num_pages_to_read * RealtimeProfilerRuntimeSizes::page_words),
        publish_batch_);
    return {.pages = num_pages_to_read, .published = published};
}

uint64_t RealtimeProfilerReceiver::run_loop(std::vector<uint32_t>& page_buf) {
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    std::chrono::microseconds backoff{1};
    uint64_t num_pages_received = 0;

    constexpr auto kFifoPlotInterval = std::chrono::milliseconds(50);
    auto last_fifo_plot = std::chrono::steady_clock::now();
    auto last_pass = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        // Recording what the clock reads took separates a stall spent blocked on PCIe from one spent anywhere else in
        // the pass. It does not attribute the remainder, and the remainder is where the surprises have been. Handed
        // to the reporter thread rather than logged: this thread blocking on a log write is itself a stall.
        if (const auto gap = now - last_pass; gap >= kDrainGapReportThreshold) {
            const auto saturate_us = [](std::chrono::nanoseconds d) {
                return std::min<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(d).count(),
                    std::numeric_limits<uint32_t>::max());
            };
            const uint64_t pack = saturate_us(gap) << 32 | saturate_us(pass_sync_busy_);
            // Single writer, so a load-then-store is a max.
            if (pack > worst_drain_gap_.load(std::memory_order_relaxed)) {
                worst_drain_gap_.store(pack, std::memory_order_relaxed);
            }
        }
        last_pass = now;
        const uint32_t num_pages = drain_all_devices(now, page_buf);
        num_pages_received += num_pages;

        if (now - last_fifo_plot >= kFifoPlotInterval) {
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler D2H FIFO high-water mark (pages)",
                static_cast<int64_t>(fifo_pages_window_max_));
            fifo_pages_window_max_ = 0;  // cleared by its only reader, the plot above
            std::chrono::nanoseconds worst_sync_error{};
            for (const auto& dev_state : devices_) {
                worst_sync_error = std::max(worst_sync_error, dev_state.clock_sync->last_published_sync_error());
            }
            TTTracyPlotD(
                RT_PROFILER,
                "RT profiler sync error (us)",
                (std::chrono::duration<double, std::micro>{worst_sync_error}.count()));
            last_fifo_plot = now;
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

uint32_t RealtimeProfilerReceiver::drain_all_devices(
    std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf) {
    uint32_t num_pages = 0;
    bool published = false;
    pass_sync_busy_ = std::chrono::nanoseconds::zero();
    for (auto& dev_state : devices_) {
        try {
            const DrainResult drained = drain_device_pages(dev_state, page_buf);
            num_pages += drained.pages;
            published |= drained.published;
            if (drained.pages != 0) {
                // Only re-read the clock after a drain that actually moved pages: an idle pass is too fast for the
                // re-read to be worth its cost.
                now = std::chrono::steady_clock::now();
            } else if (dev_state.clock_sync->due_for_probe(now)) {
                TTZoneScopedDN(RT_PROFILER, "ProbeFloor");
                TTZoneValueD(RT_PROFILER, dev_state.chip_id);
                pass_sync_busy_ += dev_state.clock_sync->resync();
            }
        } catch (const std::exception& e) {
            DeviceTelemetry& telemetry = *dev_state.telemetry;
            telemetry.drain_exceptions.fetch_add(1, std::memory_order_relaxed);
            if (telemetry.last_drain_error_mutex.try_lock()) {
                telemetry.last_drain_error = e.what();
                telemetry.last_drain_error_mutex.unlock();
            }
        }
    }
    if (published) {
        realtime_profiler_service_->wake_consumers();
    }
    return num_pages;
}

uint64_t RealtimeProfilerReceiver::drain_on_shutdown(std::vector<uint32_t>& page_buf) {
    constexpr uint32_t kShutdownDrainQuietRounds = 10;
    constexpr auto kShutdownDrainQuietBackoff = std::chrono::milliseconds(1);
    // The socket's own teardown barrier waits for the host to have acknowledged everything the device sent, so leaving
    // this loop with pages outstanding turns into a barrier timeout in the D2H socket destructor rather than anything
    // reported here. Bounded so a device that never goes quiet cannot hang teardown either.
    constexpr auto kShutdownDrainDeadline = std::chrono::seconds(5);
    const auto give_up_at = std::chrono::steady_clock::now() + kShutdownDrainDeadline;

    uint64_t num_pages_drained = 0;
    uint32_t quiet_rounds = 0;
    while (quiet_rounds < kShutdownDrainQuietRounds && std::chrono::steady_clock::now() < give_up_at) {
        const uint32_t num_pages = drain_all_devices(std::chrono::steady_clock::now(), page_buf);
        num_pages_drained += num_pages;
        bool outstanding = false;
        for (const DeviceState& dev_state : devices_) {
            outstanding = outstanding || dev_state.socket->pages_available() != 0;
        }
        if (num_pages != 0 || outstanding) {
            quiet_rounds = 0;
        } else {
            quiet_rounds++;
        }
        std::this_thread::sleep_for(kShutdownDrainQuietBackoff);
    }

    for (const DeviceState& dev_state : devices_) {
        if (const uint32_t left = dev_state.socket->pages_available(); left != 0) {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Device {} still had {} page(s) unread when the shutdown drain gave up; the D2H "
                "socket's teardown barrier will wait for them",
                dev_state.chip_id,
                left);
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
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * RealtimeProfilerRuntimeSizes::page_words);

    const uint64_t num_pages_received = run_loop(page_buf);

    // The push kernel delivers its last pages after seeing this, which the shutdown drain below collects once traffic
    // goes quiet.
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

    const uint64_t num_pages_drained = drain_on_shutdown(page_buf);

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
    // After the drain thread, so the reporter's final sweep sees everything it recorded.
    if (reporter_thread_.joinable()) {
        {
            // Under the mutex, or a notify landing between the reporter's predicate check and its block is lost and
            // this join waits out a full reporter tick.
            const std::lock_guard lock(reporter_mutex_);
            stop_.store(true, std::memory_order_release);
        }
        reporter_cv_.notify_all();
        reporter_thread_.join();
    }

    // detach_producer is idempotent, so a second shutdown() is harmless.
    realtime_profiler_service_->detach_producer(*this);

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
