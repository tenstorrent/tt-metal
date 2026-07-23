// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/realtime_profiler_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

namespace tt::tt_metal::distributed {

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

// Compute the RT-profiler L1 carve-out addresses from a base anchored past UNRESERVED (outside the user-space
// allocator).
inline RealtimeProfilerCoreL1Addrs compute_rt_profiler_core_l1_addrs(uint32_t base) {
    return {
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

size_t RealtimeProfilerManager::publish_pages(
    const DeviceState& dev_state,
    const uint32_t* page_buf,
    uint32_t num_pages,
    std::vector<tt::ProgramRealtimeRecord>& records) {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    // Each page carries a [start, end] pair of realtime_profiler_timestamp_t; the end record begins one whole record
    // in. Derive that word offset from the struct so the decode can't silently diverge from the on-wire record size.
    constexpr uint32_t kEndWord = sizeof(::realtime_profiler_timestamp_t) / sizeof(uint32_t);
    auto is_record = [](const uint32_t* page) { return page[2] != 0; };
    records.clear();
    const uint32_t chip_id = dev_state.chip_id;
    const double sync_frequency = dev_state.clock_sync.frequency();
    const tt::tt_metal::experimental::ProgramRealtimeClockSync clock_sync = dev_state.clock_sync.mapping();
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
            (static_cast<uint64_t>(rp[kEndWord]) << 32) | rp[kEndWord + 1],
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

void RealtimeProfilerManager::service_offset_servo(std::chrono::steady_clock::time_point now) {
    for (auto& dev_state : devices_) {
        dev_state.clock_sync.service_servo(now);
    }
}

RealtimeProfilerManager::RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device) :
    context_id_(mesh_device->impl().get_context_id()) {
    d2h_hugepage_fallback_ = d2h_uses_hugepage_fallback(MetalContext::instance(context_id_));
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
        // config, premature sync, corrupt state machine).
        {
            const uint32_t profiler_msg_size = factory.size_of<realtime_profiler_msgs::realtime_profiler_msg_t>();
            const uint32_t profiler_msg_words = profiler_msg_size / sizeof(uint32_t);
            std::vector<uint32_t> zero_msg(profiler_msg_words, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                device, realtime_profiler_core, realtime_profiler_base_addr, zero_msg, CoreType::WORKER);
        }

        dev_state.clock_sync.configure(
            context_id_,
            device,
            dev_state.chip_id,
            dev_state.realtime_profiler_core,
            dev_state.mesh_coord,
            d2h_hugepage_fallback_,
            realtime_profiler_base_addr + sync_host_timestamp_offset,
            mesh_device,
            realtime_profiler_base_addr + sync_ack_enc_offset,
            realtime_profiler_base_addr + sync_ack_lo_offset,
            realtime_profiler_base_addr + sync_ack_hi_offset);

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
    const auto init_throttle_now = std::chrono::steady_clock::now();
    std::vector<size_t> init_run_sync_indices;
    init_run_sync_indices.reserve(devices_.size());

    // Reuse a recent cached fit where possible (device profiler's SyncInfo masks the high word to 12 bits, shifting RT
    // zones by hours, so the profiler runs its own sync); the rest get a fresh fit below.
    for (size_t di = 0; di < devices_.size(); ++di) {
        if (!devices_[di].clock_sync.try_restore_from_cache(init_throttle_now)) {
            init_run_sync_indices.push_back(di);
        }
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
            if (dev_state.clock_sync.run_fit(100)) {
                break;
            }
        }
    });

    // First real-record offset anchor for the freshly-fitted devices (run_fit sets the slope and a rough anchor);
    // cached devices already carry a valid anchor and are skipped. The first servo tick is that anchor: it is due
    // (never anchored yet) and cannot be rejected (no fresh previous anchor to prefer).
    parallel_for_each_device_index(init_run_sync_indices, [&](size_t di) {
        std::this_thread::sleep_for(kConstructorSyncCheckDelay);
        devices_[di].clock_sync.service_servo(std::chrono::steady_clock::now());
    });
}

RealtimeProfilerManager::DrainCounts RealtimeProfilerManager::drain_device_pages(
    DeviceState& dev_state, std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf) {
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
    const size_t records = publish_pages(dev_state, page_buf.data(), num_pages_to_read, record_buf);
    return {num_pages_to_read, records};
}

uint64_t RealtimeProfilerManager::run_receiver_loop() {
    constexpr uint32_t kPageWords = RealtimeProfilerRuntimeSizes::page_size / sizeof(uint32_t);
    std::vector<uint32_t> page_buf(kMaxSocketPagesPerRead * kPageWords);
    std::vector<tt::ProgramRealtimeRecord> record_buf;
    record_buf.reserve(kMaxSocketPagesPerRead);
    constexpr std::chrono::microseconds kReceiverMaxBackoff{100};
    constexpr auto kServoPumpInterval = std::chrono::milliseconds(50);
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
        constexpr double kServoSecs = std::chrono::duration<double>(kServoPumpInterval).count();
        const uint64_t records = num_published_records_.load(std::memory_order_relaxed);
        TTTracyPlotD(
            RT_PROFILER,
            "RT profiler publish rate (rec/s)",
            static_cast<double>(records - last_diagnostics_records) / kServoSecs);
        TTTracyPlotD(RT_PROFILER, "RT profiler D2H FIFO pages", static_cast<double>(fifo_pages_window_max_));
        fifo_pages_window_max_ = 0;
        int64_t worst_sync_error_ns = 0;
        for (const auto& dev_state : devices_) {
            worst_sync_error_ns =
                std::max(worst_sync_error_ns, static_cast<int64_t>(dev_state.clock_sync.mapping().sync_error_ns));
        }
        TTTracyPlotD(RT_PROFILER, "RT sync error (us)", static_cast<double>(worst_sync_error_ns) / 1000.0);
        last_diagnostics_records = records;
    };
    emit_diagnostics_plots();
#endif
    while (!stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const bool servo_ticked = now - last_servo_tick >= kServoPumpInterval;
        if (servo_ticked) {
            last_servo_tick = now;
        }
        const uint32_t num_pages = drain_all_devices(page_buf, record_buf);
        num_pages_received += num_pages;
        if (servo_ticked) {
            service_offset_servo(now);
        }
#if defined(TRACY_ENABLE) && TT_TRACY_CATEGORY_RT_PROFILER
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
    std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf) {
    // Wake consumers only when records were actually published, not merely when pages were drained: a page that fails
    // the id filter (is_record) publishes nothing, so waking on raw page count could wake consumers with nothing new.
    uint32_t num_pages = 0;
    size_t records_published = 0;
    for (auto& dev_state : devices_) {
        try {
            const DrainCounts counts = drain_device_pages(dev_state, page_buf, record_buf);
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
        const uint32_t num_pages = drain_all_devices(page_buf, record_buf);
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

    // Signal the device kernels to terminate only after the servo loop above has exited, so no handshake fires against
    // a core that is already tearing down (that would just time out). The push kernel then delivers its last pages,
    // which the shutdown drain below collects once traffic goes quiet.
    for (auto& dev_state : devices_) {
        if (dev_state.core_l1.ring_buffer == 0 || !dev_state.device) {
            continue;
        }
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

    const uint64_t num_pages_drained = drain_receiver_on_shutdown();

    log_debug(
        tt::LogMetal,
        "[Real-time profiler] Receiver thread stopped after {} pages ({} drained during shutdown)",
        num_pages_received + num_pages_drained,
        num_pages_drained);
}

RealtimeProfilerManager::~RealtimeProfilerManager() { shutdown(); }

void RealtimeProfilerManager::shutdown() {
    // The receiver thread signals terminate to the device kernels and drains their final pages on its way out (see
    // run_receiver), so stopping it is all shutdown needs to do here.
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

}  // namespace tt::tt_metal::distributed
