// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/device_programs.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdlib>
#include <future>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include <enchantum/enchantum.hpp>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/sync/link_sync.hpp"
#include "impl/streaming_profiler/sync/tile_sync.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

uint32_t packed_xy(const CoreCoord& core) {
    return kernel_profiler::word_of(
        kernel_profiler::NocXy{.x = static_cast<uint32_t>(core.x), .y = static_cast<uint32_t>(core.y)});
}

// The DRAM view each relay runs on, in relay order. A relay's position in this order picks its band of the worker grid
// and its write VC.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
// Two pairs of staging slots, so one pair can gather a frame while the other is sent, plus three that spool mode splits
// into two bounce buffers.
constexpr uint32_t kStageSlots = 7;
// The most cores one relay drains. A relay keeps one record per core, and 72 lets two relays split a 140-core grid.
constexpr uint32_t kMaxRelayCores = 72;
constexpr uint32_t kCoreRecordsBytes = kMaxRelayCores * kernel_profiler::kRelayCoreRecordBytes;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint32_t kLanesPerTile = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
// The smallest host FIFO the eth relay's sync socket gets, large enough to ride out multi-second page-fault stalls on
// the sync thread.
constexpr uint32_t kSyncMinFifoBytes = 128u << 20;
constexpr uint32_t kEthSyncRingBytes = kernel_profiler::kSyncRingRecords * sizeof(kernel_profiler::SyncRecord);
// The bytes reserved for a resident core's control block. It is one 64 B block, so the eth control block stays aligned
// for the eth relay's single 64 B NoC read of it.
constexpr uint32_t kCtrlBytes = 64;
static_assert(sizeof(kernel_profiler::ResidentCtrl) <= kCtrlBytes);
constexpr uint32_t kArmedOffset = kernel_profiler::PROFILER_ARMED * sizeof(uint32_t);
constexpr uint32_t kLinkCtlOffset = kernel_profiler::SPSC_LINK_SYNC_CTL * sizeof(uint32_t);
constexpr uint32_t kLinkDoneOffset = kernel_profiler::SPSC_LINK_SYNC_DONE * sizeof(uint32_t);
constexpr auto kResidentTimeout = std::chrono::seconds(10);
constexpr auto kHeartbeatTimeout = std::chrono::milliseconds(500);
constexpr auto kLinkRoundsTimeout = std::chrono::seconds(1);

void write_u32(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t value) {
    cluster.write_core(&value, sizeof(value), tt_cxy_pair(chip, virt), addr);
}

template <std::predicate<uint32_t> Pred>
[[nodiscard]] bool poll_word(
    tt::Cluster& cluster, const tt_cxy_pair& core, uint64_t addr, std::chrono::milliseconds timeout, Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        uint32_t word = 0;
        cluster.read_core(&word, sizeof(word), core, addr);
        if (pred(word)) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

KernelHandle create_idle_eth_kernel(
    Program& program,
    const std::string& src,
    const CoreCoord& core,
    DataMovementProcessor processor,
    std::unordered_map<std::string, uint32_t> named_args) {
    return CreateKernel(
        program,
        src,
        core,
        EthernetConfig{
            .eth_mode = Eth::IDLE,
            .noc = processor == DataMovementProcessor::RISCV_0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
            .processor = processor,
            .named_compile_args = std::move(named_args)});
}

void compile_resident(IDevice* device, Program& program) {
    program.impl().compile(device, /*force_slow_dispatch=*/true);
    slow_dispatch::WriteRuntimeArgsToDevice(*device, program, /*force_slow_dispatch=*/true);
}

}  // namespace

bool can_capture(const Hal& hal, const llrt::RunTimeOptions& rtoptions) {
    return rtoptions.get_streaming_profiler_enabled() && hal.get_arch() == tt::ARCH::BLACKHOLE &&
           hal.has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

bool can_capture(const MetalContext& mc) { return can_capture(mc.hal(), mc.rtoptions()); }

CoreCoords locate_core(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type) {
    CoreCoord phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true);
    if (type == CoreType::DRAM) {
        const tt::umd::CoreCoord noc0 = cluster.get_soc_desc(chip).translate_coord_to(
            tt::umd::CoreCoord(phys.x, phys.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
        phys = CoreCoord(noc0.x, noc0.y);
    }
    return CoreCoords{
        .logical = logical,
        .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
        .phys = phys};
}

void zero_l1(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t bytes) {
    const std::vector<uint8_t> zeros(bytes);
    cluster.write_core(zeros.data(), bytes, tt_cxy_pair(chip, virt), addr);
}

void zero_profiler_control(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr) {
    zero_l1(cluster, chip, virt, addr, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
}

void launch_resident(IDevice* device, Program& program) {
    compile_resident(device, program);
    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/true);
}

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> out(cores.begin(), cores.end());
    std::ranges::sort(out, {}, [](const CoreCoord& core) { return std::pair(core.y, core.x); });
    return out;
}

DevicePrograms::DeviceCtx::DeviceCtx(distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord) :
    mesh(&mesh), coord(coord), device(mesh.get_device(coord)) {}
DevicePrograms::DeviceCtx::~DeviceCtx() = default;
DevicePrograms::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

DevicePrograms::DevicePrograms() = default;

DevicePrograms::~DevicePrograms() { quiesce({}); }

void DevicePrograms::carve_l1() {
    const Hal& hal = mc_->hal();
    // One D2H socket's config, rounded up to whole 64 B blocks so the eth control block below the configs stays 64 B
    // aligned.
    const uint32_t cfg_bytes =
        tt::align(distributed::D2HSocket::required_config_buffer_size(hal.get_alignment(HalMemType::L1)), kCtrlBytes);
    const uint32_t slot_bytes = kernel_profiler::spsc_span_slot_words(kLanesPerTile) * sizeof(uint32_t);
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    drisc_l1_.stage_base = base;
    drisc_l1_.core_records = drisc_l1_.stage_base + kStageSlots * slot_bytes;
    drisc_l1_.ctrl = drisc_l1_.core_records + kCoreRecordsBytes;
    drisc_l1_.cfg = base + region - cfg_bytes;
    drisc_l1_.host_ctrl = drisc_l1_.ctrl + hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    TT_FATAL(
        drisc_l1_.ctrl + kCtrlBytes <= drisc_l1_.cfg,
        "streaming profiler: DRISC L1 ({} B unreserved) cannot hold a relay's {} staging slots, core records and "
        "socket config",
        region,
        kStageSlots);

    const uint32_t eth_base = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_size = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    eth_l1_.frames_cfg = eth_base + eth_size - 2 * cfg_bytes;
    eth_l1_.sync_cfg = eth_l1_.frames_cfg + cfg_bytes;
    eth_l1_.ctrl = eth_l1_.frames_cfg - kCtrlBytes;
    // spsc_span_pack_pad() assumes the staged frame starts on a page boundary.
    eth_l1_.stage = (eth_l1_.ctrl - slot_bytes) & ~(kPageBytes - 1u);
    eth_l1_.sample_ring = (eth_l1_.stage - sizeof(kernel_profiler::SyncSampleRing)) & ~(kPageBytes - 1u);
    eth_l1_.sync_ring = eth_l1_.sample_ring - kEthSyncRingBytes;
    eth_l1_.link = link_sync::l1_addr(hal);
    eth_l1_.link_ring = eth_l1_.link + offsetof(kernel_profiler::LinkSyncL1, ring);
    TT_FATAL(
        eth_l1_.sync_ring >= eth_base,
        "streaming profiler: idle-eth L1 too small for the wall-clock core ({} B unreserved, {} B needed)",
        eth_size,
        eth_base + eth_size - eth_l1_.sync_ring);

    const uint32_t spool_mb = mc_->rtoptions().get_streaming_profiler_spool_mb();
    if (spool_mb != 0) {
        drisc_l1_.spool_addr = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
        drisc_l1_.spool_bytes = spool_mb << 20;
    }
}

bool DevicePrograms::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    mc_ = &MetalContext::instance(mesh_device->impl().get_context_id());
    auto& cluster = mc_->get_cluster();
    const auto& hal = mc_->hal();
    const auto& rtopts = mc_->rtoptions();
    if (!can_capture(*mc_)) {
        log_warning(
            tt::LogMetal, "[streaming profiler] not capturing: it needs Blackhole with DRAM programmable cores");
        return false;
    }
    fabric_link_sync_ = mc_->get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
    capture_.sync_check = rtopts.get_streaming_profiler_sync_check_enabled();
    control_vector_l1_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    carve_l1();
    const uint32_t relay_fifo_bytes = rtopts.get_streaming_profiler_fifo_mb() << 20;
    const uint32_t sync_fifo_bytes = std::max(relay_fifo_bytes, kSyncMinFifoBytes);

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx& ctx = devices_.emplace_back(*mesh_device, coord);
        ctx.index = static_cast<uint32_t>(devices_.size() - 1);
        ctx.chip_id = static_cast<uint32_t>(ctx.device->id());
        ctx.numa_node = static_cast<int>(cluster.get_numa_node_for_device(ctx.chip_id));
        capture_.devices.emplace_back().chip_id = ctx.chip_id;
        enumerate_worker_grid(ctx);
        enumerate_eth_cores(ctx);
        choose_relay_cores(ctx);

        CaptureContext::Device& capture = capture_.devices[ctx.index];
        const TileClocks* clocks = tile_clocks(mc_->get_context_id(), ctx.chip_id);
        TT_FATAL(clocks != nullptr, "streaming profiler: device {} has no tile clocks", ctx.chip_id);
        const auto clock_of = [&](CoreType type, const CoreCoord& logical) {
            const auto it = clocks->find({type, logical});
            TT_FATAL(
                it != clocks->end(),
                "streaming profiler: device {} has no tile clock for {} core {}",
                ctx.chip_id,
                enchantum::to_string(type),
                logical.str());
            return it->second;
        };
        const int64_t wall_clock_offset = clock_of(CoreType::ETH, ctx.wall_clock.core.logical);
        for (size_t i = 0; i < ctx.producers.size(); i++) {
            const Producer& producer = ctx.producers[i];
            capture.clock_offsets[i] = wall_clock_offset - clock_of(producer.type, producer.logical);
        }
        if (ctx.check) {
            capture.check_offset = wall_clock_offset - clock_of(CoreType::ETH, ctx.check->core.logical);
        }
    }
    if (devices_.empty()) {
        log_warning(tt::LogMetal, "[streaming profiler] not capturing: no device of this mesh is local to this host");
        return false;
    }
    const auto start_device = [&](DeviceCtx& ctx) {
        for (uint32_t relay_index = 0; relay_index < ctx.relays.size(); relay_index++) {
            start_resident(
                ctx,
                ctx.relays[relay_index],
                HalProgrammableCoreType::DRAM,
                {{.cfg = drisc_l1_.cfg, .fifo_bytes = relay_fifo_bytes}},
                relay_program(ctx, relay_index));
        }
        start_resident(ctx, ctx.wall_clock, HalProgrammableCoreType::IDLE_ETH, {}, wall_clock_program(ctx));
        if (ctx.check) {
            start_resident(ctx, *ctx.check, HalProgrammableCoreType::IDLE_ETH, {}, check_program(ctx));
        }
        start_resident(
            ctx,
            ctx.eth_relay,
            HalProgrammableCoreType::IDLE_ETH,
            {{.cfg = eth_l1_.sync_cfg, .fifo_bytes = sync_fifo_bytes, .sync_socket = true},
             {.cfg = eth_l1_.frames_cfg, .fifo_bytes = relay_fifo_bytes}},
            eth_relay_program(ctx));
        write_producers_armed(ctx, 1);
    };
    std::vector<std::future<void>> starts;
    starts.reserve(devices_.size());
    for (DeviceCtx& ctx : devices_) {
        starts.push_back(std::async(std::launch::async, start_device, std::ref(ctx)));
    }
    for (auto& start : starts) {
        start.get();
    }
    for (DeviceCtx& ctx : devices_) {
        std::ranges::move(ctx.captured, std::back_inserter(sockets_));
        ctx.captured.clear();
    }
    plan_links();
    log_info(
        tt::LogMetal,
        "[streaming profiler] active on {} device(s){}",
        devices_.size(),
        rtopts.get_streaming_profiler_tracy_enabled() ? " with the Tracy sink" : "");
    return true;
}

void DevicePrograms::start_resident(
    DeviceCtx& ctx,
    ResidentCore& resident,
    HalProgrammableCoreType core_type,
    const std::vector<SocketSpec>& specs,
    std::unique_ptr<Program> program) {
    auto& cluster = mc_->get_cluster();
    if (resident.stale_l1) {
        zero_l1(cluster, ctx.chip_id, resident.core.virt, resident.stale_l1->addr, resident.stale_l1->bytes);
    }
    resident.socket_index = static_cast<uint32_t>(ctx.sockets.size());
    resident.socket_count = static_cast<uint32_t>(specs.size());
    for (const SocketSpec& spec : specs) {
        const auto& socket = ctx.sockets.emplace_back(std::make_unique<distributed::D2HSocket>(
            ctx.mesh->shared_from_this(),
            distributed::MeshCoreCoord{ctx.coord, resident.core.phys},
            spec.fifo_bytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = spec.cfg, .sender_core_type = core_type},
            distributed::D2HSocket::ProcessScope::InProcess));
        socket->set_page_size(kPageBytes);
        ctx.captured.push_back(CapturedSocket{
            .socket = socket.get(),
            .device_index = ctx.index,
            .socket_index = static_cast<uint32_t>(ctx.sockets.size() - 1),
            .numa_node = ctx.numa_node,
            .sync_socket = spec.sync_socket});
    }
    // A previous capture's done, heartbeat and stop words would otherwise be read as this capture's.
    zero_l1(cluster, ctx.chip_id, resident.core.virt, resident.ctrl, kCtrlBytes);
    launch_resident(ctx.device, *program);
    // A resident kernel is launched without waiting for it, so a core stuck in reset would report nothing. Two
    // heartbeats show that its loop is running.
    uint32_t heartbeat = 0;
    const bool started = poll_word(
        cluster,
        tt_cxy_pair(ctx.chip_id, resident.core.virt),
        resident.ctrl + offsetof(kernel_profiler::ResidentCtrl, heartbeat),
        kHeartbeatTimeout,
        [&](uint32_t word) {
            heartbeat = word;
            return word >= 2;
        });
    TT_FATAL(
        started,
        "streaming profiler: device {} {} did not start (heartbeat {} after launch)",
        ctx.chip_id,
        resident.name,
        heartbeat);
    resident.program = std::move(program);
}

DevicePrograms::Producer& DevicePrograms::add_producer(
    DeviceCtx& ctx, const CoreCoords& core, CoreType type, uint64_t control_vector_l1) {
    using experimental::streaming_profiler::Processor;
    // An eth core has two RISCs, so its other lanes never carry a record.
    constexpr std::array<Processor, kLanesPerTile> kTensix = {
        Processor::BRISC, Processor::NCRISC, Processor::TRISC0, Processor::TRISC1, Processor::TRISC2};
    constexpr std::array<Processor, kLanesPerTile> kEth = {
        Processor::ERISC0, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1};
    CaptureContext::Device& capture = capture_.devices[ctx.index];
    capture.core_of_xy[packed_xy(core.virt)] = static_cast<uint16_t>(capture.clock_offsets.size());
    capture.clock_offsets.push_back(0);
    for (const Processor processor : type == CoreType::ETH ? kEth : kTensix) {
        capture.lanes.push_back(experimental::streaming_profiler::Core{
            .logical = core.logical,
            .physical = core.phys,
            .chip_id = static_cast<ChipId>(ctx.chip_id),
            .processor = processor});
    }
    return ctx.producers.emplace_back(Producer{core, type, control_vector_l1});
}

void DevicePrograms::enumerate_worker_grid(DeviceCtx& ctx) {
    auto& cluster = mc_->get_cluster();
    const uint32_t chip = ctx.chip_id;
    // A core whose ring nobody drains fills it and hangs the host at close, so every compute core is a producer.
    const CoreCoord grid = ctx.mesh->compute_with_storage_grid_size();
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            const Producer& producer = add_producer(
                ctx,
                locate_core(cluster, chip, CoreCoord{lx, ly}, CoreType::WORKER),
                CoreType::WORKER,
                control_vector_l1_);
            zero_profiler_control(cluster, chip, producer.virt, producer.control_vector_l1);
        }
    }
    ctx.worker_count = static_cast<uint32_t>(ctx.producers.size());

    // Every Tensix core's armed flag is cleared, because a dispatch core's ring is never drained and its L1 survives
    // re-init, so an armed flag left from an earlier capture would block it on a full ring and hang close.
    const CoreCoord tensix_grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
    for (uint32_t ly = 0; ly < tensix_grid.y; ly++) {
        for (uint32_t lx = 0; lx < tensix_grid.x; lx++) {
            const CoreCoord virt =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord{lx, ly}, CoreType::WORKER);
            write_u32(cluster, chip, virt, control_vector_l1_ + kArmedOffset, 0);
        }
    }
}

void DevicePrograms::choose_relay_cores(DeviceCtx& ctx) {
    auto& cluster = mc_->get_cluster();
    const uint32_t chip = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(chip);
    const uint32_t view_count = static_cast<uint32_t>(soc.get_num_dram_views());
    std::vector<uint32_t> banks;
    std::ranges::copy_if(kRelayBankRoster, std::back_inserter(banks), [&](uint32_t bank) { return bank < view_count; });
    ctx.relays.resize(banks.size());
    for (uint32_t relay_index = 0; relay_index < ctx.relays.size(); relay_index++) {
        ctx.relays[relay_index].core = locate_core(
            cluster,
            chip,
            ctx.mesh->impl().pick_unused_dram_logical_core(ctx.device, banks[relay_index]),
            CoreType::DRAM);
    }
    // pick_unused_dram_logical_core() can't see two views resolving to one physical port, and two relays on one L1
    // would overlap.
    for (uint32_t a = 0; a < ctx.relays.size(); a++) {
        for (uint32_t b = a + 1; b < ctx.relays.size(); b++) {
            TT_FATAL(
                ctx.relays[a].core.virt != ctx.relays[b].core.virt,
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to DRAM core {}. "
                "Two resident relay kernels cannot share a core.",
                a,
                banks[a],
                b,
                banks[b],
                ctx.relays[a].core.virt.str());
        }
    }
    // A DRAM channel split into several views has its own set of preferred endpoints for each view, so a subchannel
    // that is free in one view can be a preferred endpoint of another.
    for (uint32_t relay_index = 0; relay_index < ctx.relays.size(); relay_index++) {
        ResidentCore& relay = ctx.relays[relay_index];
        const uint8_t noc2axi_mask =
            soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(relay.core.logical));
        TT_FATAL(
            noc2axi_mask == 0,
            "streaming profiler: device {} relay {}'s DRISC ({},{}) is a DRAM view's preferred endpoint on NOC mask "
            "{:#x}, so firmware holds those NIUs in NOC2AXI mode and the relay cannot initiate NoC traffic on them",
            chip,
            relay_index,
            relay.core.virt.x,
            relay.core.virt.y,
            noc2axi_mask);
        relay.name = fmt::format("relay {}", relay_index);
        relay.ctrl = drisc_l1_.host_ctrl;
        // The relay is built with PROFILE_KERNEL and nothing drains its own profiler ring, which fills after about 74
        // launches since the last board reset and hangs the RISC in firmware init.
        relay.stale_l1 = L1Range{
            .addr = mc_->hal().get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER),
            .bytes = kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE};
    }
}

std::unique_ptr<Program> DevicePrograms::relay_program(const DeviceCtx& ctx, uint32_t relay_index) {
    const ResidentCore& relay = ctx.relays[relay_index];
    const uint32_t num_cores = ctx.worker_count;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * relay_index) / ctx.relays.size());
    const uint32_t hi =
        static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * (relay_index + 1)) / ctx.relays.size());
    const uint32_t my_cores = hi - lo;
    TT_FATAL(
        my_cores <= kMaxRelayCores,
        "streaming profiler: device {} relay {} would own {} cores but a relay holds at most {}; this {}-core grid "
        "needs {} relays and the part has {}",
        ctx.chip_id,
        relay_index,
        my_cores,
        kMaxRelayCores,
        num_cores,
        tt::div_up(num_cores, kMaxRelayCores),
        ctx.relays.size());
    auto program = std::make_unique<Program>(CreateProgram());
    const KernelHandle kernel = CreateKernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/drisc_relay.cpp",
        relay.core.logical,
        // Writes take about twice as long on NOC 1, so a relay there would cause nearly every producer stall.
        DramConfig{
            .noc = NOC::NOC_0,
            .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}},
            .named_compile_args = {
                {"stage_base", drisc_l1_.stage_base},
                {"n_stage", kStageSlots},
                {"core_records", drisc_l1_.core_records},
                {"ctrl", drisc_l1_.ctrl},
                {"socket_config_addr", drisc_l1_.cfg},
                {"max_cores", kMaxRelayCores},
                // Bit 1 of the relay index splits the relays' pushes between two of the four unicast request VCs.
                {"write_vc", (relay_index & 2u) ? 0u : 1u},
                {"spool_base", drisc_l1_.spool_addr},
                {"spool_bytes", drisc_l1_.spool_bytes}}});
    std::vector<uint32_t> runtime_args = {my_cores, static_cast<uint32_t>(control_vector_l1_)};
    // Reversed so the last-launched cores get the first-serviced slots.
    std::ranges::copy(
        std::span(ctx.producers).subspan(lo, my_cores) | std::views::reverse |
            std::views::transform([](const Producer& producer) { return packed_xy(producer.virt); }),
        std::back_inserter(runtime_args));
    SetRuntimeArgs(*program, kernel, relay.core.logical, runtime_args);
    return program;
}

void DevicePrograms::enumerate_eth_cores(DeviceCtx& ctx) {
    auto& cluster = mc_->get_cluster();
    const auto& hal = mc_->hal();
    const uint32_t chip = ctx.chip_id;
    const std::vector<CoreCoord> idle = sorted_yx(ctx.device->get_inactive_ethernet_cores());
    TT_FATAL(
        idle.size() >= 2,
        "streaming profiler: device {} has {} idle ethernet cores; the wall-clock core and its eth relay need two",
        chip,
        idle.size());
    std::vector<CoreCoords> idle_eth;
    std::ranges::transform(idle, std::back_inserter(idle_eth), [&](const CoreCoord& logical) {
        return locate_core(cluster, chip, logical, CoreType::ETH);
    });
    ctx.wall_clock = {
        .core = idle_eth.front(),
        .name = "wall clock",
        .ctrl = eth_l1_.ctrl,
        .stale_l1 = L1Range{.addr = eth_l1_.sample_ring, .bytes = offsetof(kernel_profiler::SyncSampleRing, samples)}};
    const uint64_t idle_eth_control_vector_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::PROFILER);
    const Producer& wall_clock = add_producer(ctx, ctx.wall_clock.core, CoreType::ETH, idle_eth_control_vector_l1);
    zero_profiler_control(cluster, chip, wall_clock.virt, wall_clock.control_vector_l1);
    const auto nearest = [&](const CoreCoord& target) -> std::optional<CoreCoords> {
        auto others = idle_eth | std::views::drop(1) |
                      std::views::filter([&](const CoreCoords& other) { return other.phys != target; });
        const auto it = std::ranges::min_element(others, {}, [&](const CoreCoords& other) {
            return static_cast<uint32_t>(std::abs(static_cast<int>(other.phys.x) - static_cast<int>(target.x))) +
                   static_cast<uint32_t>(std::abs(static_cast<int>(other.phys.y) - static_cast<int>(target.y)));
        });
        return it == others.end() ? std::nullopt : std::optional(*it);
    };
    ctx.eth_relay = {.core = *nearest(ctx.wall_clock.core.phys), .name = "idle-eth relay", .ctrl = eth_l1_.ctrl};
    if (capture_.sync_check) {
        const std::optional<CoreCoords> check = nearest(ctx.eth_relay.core.phys);
        TT_FATAL(
            check.has_value(),
            "streaming profiler: device {} has {} idle ethernet cores; the check core needs a third",
            chip,
            idle_eth.size());
        ctx.check = ResidentCore{.core = *check, .name = "sync check", .ctrl = eth_l1_.ctrl};
    }
    const uint64_t active_eth_control_vector_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
    for (const CoreCoord& logical :
         sorted_yx(ctx.device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
        const Producer& producer = add_producer(
            ctx, locate_core(cluster, chip, logical, CoreType::ETH), CoreType::ETH, active_eth_control_vector_l1);
        // When the fabric routers run, one is already writing into this ring, so its control block is left alone.
        if (!fabric_link_sync_) {
            zero_profiler_control(cluster, chip, producer.virt, producer.control_vector_l1);
        }
    }
    for (const CoreCoords& idle_core : idle_eth | std::views::drop(1)) {
        if (idle_core.phys != ctx.eth_relay.core.phys && (!ctx.check || idle_core.phys != ctx.check->core.phys)) {
            const Producer& producer = add_producer(ctx, idle_core, CoreType::ETH, idle_eth_control_vector_l1);
            zero_profiler_control(cluster, chip, producer.virt, producer.control_vector_l1);
        }
    }
}

std::unique_ptr<Program> DevicePrograms::wall_clock_program(const DeviceCtx& ctx) {
    const CoreCoord& core = ctx.wall_clock.core.logical;
    auto program = std::make_unique<Program>(CreateProgram());
    create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampler.cpp",
        core,
        DataMovementProcessor::RISCV_0,
        {{"ctrl", eth_l1_.ctrl}, {"sample_ring", eth_l1_.sample_ring}});
    create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_model.cpp",
        core,
        DataMovementProcessor::RISCV_1,
        {{"ctrl", eth_l1_.ctrl}, {"sync_ring", eth_l1_.sync_ring}, {"sample_ring", eth_l1_.sample_ring}});
    return program;
}

std::unique_ptr<Program> DevicePrograms::check_program(const DeviceCtx& ctx) {
    auto program = std::make_unique<Program>(CreateProgram());
    create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_check.cpp",
        ctx.check->core.logical,
        DataMovementProcessor::RISCV_0,
        {{"ctrl", eth_l1_.ctrl}, {"sync_ring", eth_l1_.sync_ring}});
    return program;
}

std::unique_ptr<Program> DevicePrograms::eth_relay_program(const DeviceCtx& ctx) {
    auto program = std::make_unique<Program>(CreateProgram());
    const Producer& wall_clock = ctx.wall_clock_producer();
    const KernelHandle kernel = create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_relay.cpp",
        ctx.eth_relay.core.logical,
        DataMovementProcessor::RISCV_0,
        {{"frames_cfg", eth_l1_.frames_cfg},
         {"sync_cfg", eth_l1_.sync_cfg},
         {"stage", eth_l1_.stage},
         {"ctrl", eth_l1_.ctrl},
         {"wall_clock_xy", packed_xy(wall_clock.virt)},
         {"wall_clock_control_vector_l1", static_cast<uint32_t>(wall_clock.control_vector_l1)},
         {"sync_ring", eth_l1_.sync_ring},
         {"link_ring", eth_l1_.link_ring},
         {"sync_check", ctx.check ? 1u : 0u},
         {"check_xy", ctx.check ? packed_xy(ctx.check->core.virt) : 0u}});
    const auto drained = ctx.drained_by_eth_relay();
    const auto drained_count = static_cast<uint32_t>(drained.size());
    TT_FATAL(
        drained_count <= kernel_profiler::kEthRelayMaxDrained,
        "streaming profiler: device {} has {} eth cores to drain; the eth relay holds at most {}",
        ctx.chip_id,
        drained_count,
        kernel_profiler::kEthRelayMaxDrained);
    std::vector<uint32_t> runtime_args = {drained_count};
    for (const Producer& producer : drained) {
        runtime_args.push_back(packed_xy(producer.virt));
        runtime_args.push_back(static_cast<uint32_t>(producer.control_vector_l1));
    }
    SetRuntimeArgs(*program, kernel, ctx.eth_relay.core.logical, runtime_args);
    return program;
}

void DevicePrograms::write_producers_armed(const DeviceCtx& ctx, uint32_t armed) {
    auto& cluster = mc_->get_cluster();
    for (const Producer& producer : ctx.producers) {
        write_u32(cluster, ctx.chip_id, producer.virt, producer.control_vector_l1 + kArmedOffset, armed);
    }
}

void DevicePrograms::plan_links() {
    const tt_fabric::ControlPlane* control_plane = fabric_link_sync_ ? &mc_->get_control_plane() : nullptr;
    // The eth relay drains every active eth core, so every port's core is among them.
    const auto core_of = [&](size_t dev, const CoreCoord& eth) {
        const DeviceCtx& ctx = devices_[dev];
        const auto drained = ctx.drained_by_eth_relay();
        return static_cast<uint32_t>(
            std::to_address(std::ranges::find(drained, eth, &Producer::logical)) - ctx.producers.data());
    };
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            const bool flip = chip_a > devices_[b].chip_id;
            for (const link_sync::Link& link :
                 link_sync::links_between(mc_->get_cluster(), control_plane, chip_a, devices_[b].chip_id)) {
                const size_t dev_a = flip ? b : a, dev_b = flip ? a : b;
                capture_.links.push_back(CaptureContext::Link{
                    .dev_a = static_cast<uint32_t>(dev_a),
                    .dev_b = static_cast<uint32_t>(dev_b),
                    .core_a = core_of(dev_a, link.eth_a),
                    .core_b = core_of(dev_b, link.eth_b),
                    .eth_a = link.eth_a,
                    .eth_b = link.eth_b});
            }
        }
    }
    const std::vector<bool> reached = reached_from_root(capture_.links, devices_.size(), [](size_t) { return true; });
    for (size_t dev = 0; dev < devices_.size(); dev++) {
        TT_FATAL(
            reached[dev],
            "streaming profiler: no link path from device {} to the root device {}",
            devices_[dev].chip_id,
            devices_[CaptureContext::kRootDevice].chip_id);
    }
}

std::array<DevicePrograms::LinkPort, 2> DevicePrograms::link_ports(const CaptureContext::Link& link) const {
    const auto port = [&](uint32_t dev, uint32_t core) {
        return LinkPort{.ctx = devices_[dev], .producer = devices_[dev].producers[core]};
    };
    return {port(link.dev_a, link.core_a), port(link.dev_b, link.core_b)};
}

void DevicePrograms::launch_links() {
    auto& cluster = mc_->get_cluster();
    links_running_ = true;
    if (!fabric_link_sync_) {
        for (const CaptureContext::Link& link : capture_.links) {
            const std::array ports = link_ports(link);
            for (const LinkPort& port : ports) {
                link_sync::zero_port(mc_->hal(), cluster, port.ctx.chip_id, port.producer.virt);
            }
            // Both are compiled first, because the transmitter starts its handshake as soon as it runs.
            std::array<std::unique_ptr<Program>, 2> programs;
            for (size_t i = 0; i < ports.size(); i++) {
                programs[i] = std::make_unique<Program>(CreateProgram());
                CreateKernel(
                    *programs[i],
                    "tt_metal/impl/streaming_profiler/kernels/link_sync.cpp",
                    ports[i].producer.logical,
                    EthernetConfig{
                        .noc = NOC::RISCV_0_default,
                        .named_compile_args = link_sync::compile_args(
                            i == 0 ? kernel_profiler::LinkSyncRole::Transmitter
                                   : kernel_profiler::LinkSyncRole::Receiver,
                            eth_l1_.link,
                            capture_.sync_check)});
                compile_resident(ports[i].ctx.device, *programs[i]);
            }
            for (size_t i = 0; i < ports.size(); i++) {
                slow_dispatch::LaunchProgramAsync(*ports[i].ctx.device, *programs[i], /*force_slow_dispatch=*/true);
            }
            std::ranges::move(programs, std::back_inserter(resident_link_programs_));
        }
    }
    // Every transmitter, resident or in a fabric router, sends nothing until its ctl word is Run.
    for (const CaptureContext::Link& link : capture_.links) {
        const LinkPort transmitter = link_ports(link)[0];
        write_u32(
            cluster,
            transmitter.ctx.chip_id,
            transmitter.producer.virt,
            transmitter.producer.control_vector_l1 + kLinkCtlOffset,
            static_cast<uint32_t>(kernel_profiler::LinkSyncCtl::Run));
    }
}

void DevicePrograms::stop_links() {
    if (!links_running_) {
        return;
    }
    auto& cluster = mc_->get_cluster();
    // Each port must record two solved rounds before any port stops, because the clock solver needs two rounds to solve
    // a link and a port only records a round when the next one starts. Rounds count from 1, and under the sync check
    // only the rounds numbered a multiple of kLinkSyncCheckSolveEvery are solved.
    const uint32_t solve_every = capture_.sync_check ? kernel_profiler::kLinkSyncCheckSolveEvery : 1;
    const uint32_t rounds_needed = 2 * solve_every;
    for (const CaptureContext::Link& link : capture_.links) {
        for (const LinkPort& port : link_ports(link)) {
            const bool recorded = poll_word(
                cluster,
                tt_cxy_pair(port.ctx.chip_id, port.producer.virt),
                port.producer.control_vector_l1 + kernel_profiler::SPSC_LINK_SYNC_TAIL * sizeof(uint32_t),
                kLinkRoundsTimeout,
                [&](uint32_t tail) { return tail >= kernel_profiler::kLinkSyncRecordsPerRound * rounds_needed; });
            TT_FATAL(
                recorded,
                "streaming profiler: device {} link sync port {} recorded fewer than {} rounds within {}, so its "
                "link is dead",
                port.ctx.chip_id,
                port.producer.virt.str(),
                rounds_needed,
                kLinkRoundsTimeout);
        }
    }
    for (const CaptureContext::Link& link : capture_.links) {
        for (const LinkPort& port : link_ports(link)) {
            write_u32(
                cluster,
                port.ctx.chip_id,
                port.producer.virt,
                port.producer.control_vector_l1 + kLinkCtlOffset,
                static_cast<uint32_t>(kernel_profiler::LinkSyncCtl::Stop));
            const bool stopped = poll_word(
                cluster,
                tt_cxy_pair(port.ctx.chip_id, port.producer.virt),
                port.producer.control_vector_l1 + kLinkDoneOffset,
                kLinkRoundsTimeout,
                [](uint32_t word) { return word != 0; });
            TT_FATAL(
                stopped,
                "streaming profiler: device {} link sync port {} did not stop within {}",
                port.ctx.chip_id,
                port.producer.virt.str(),
                kLinkRoundsTimeout);
        }
    }
    resident_link_programs_.clear();
}

void DevicePrograms::start() {
    auto& cluster = mc_->get_cluster();
    const uint64_t go_addr = eth_l1_.ctrl + offsetof(kernel_profiler::ResidentCtrl, go);
    for (const DeviceCtx& ctx : devices_) {
        write_u32(cluster, ctx.chip_id, ctx.wall_clock.core.virt, go_addr, 1);
        write_u32(cluster, ctx.chip_id, ctx.eth_relay.core.virt, go_addr, 1);
    }
    // The check core starts after the wall-clock core has published its first clock point, so every check reading has a
    // clock point before it.
    for (const DeviceCtx& ctx :
         devices_ | std::views::filter([](const DeviceCtx& device_ctx) { return device_ctx.check.has_value(); })) {
        const bool published = poll_word(
            cluster,
            tt_cxy_pair(ctx.chip_id, ctx.wall_clock.core.virt),
            eth_l1_.ctrl + offsetof(kernel_profiler::ResidentCtrl, sync.tail),
            kResidentTimeout,
            [](uint32_t word) { return word != 0; });
        TT_FATAL(
            published,
            "streaming profiler: device {} wall-clock core {} published no clock point within {}",
            ctx.chip_id,
            ctx.wall_clock.core.virt.str(),
            kResidentTimeout);
        write_u32(cluster, ctx.chip_id, ctx.check->core.virt, go_addr, 1);
    }
    launch_links();
}

void DevicePrograms::await_stop(const DeviceCtx& ctx, const ResidentCore& resident, const RelayStateFn& on_state) {
    const auto sockets = std::span(ctx.sockets).subspan(resident.socket_index, resident.socket_count);
    const auto report = [&](RelayState state) {
        for (uint32_t k = 0; k < resident.socket_count; k++) {
            on_state(ctx.index, resident.socket_index + k, state);
        }
    };
    auto& cluster = mc_->get_cluster();
    bool awaiting_acks = false;
    uint32_t done_word = 0;
    const bool done = poll_word(
        cluster,
        tt_cxy_pair(ctx.chip_id, resident.core.virt),
        resident.ctrl + offsetof(kernel_profiler::ResidentCtrl, done),
        kResidentTimeout,
        [&](uint32_t word) {
            // With no consumer, the pages are discarded so the relay's barriers still complete.
            if (!on_state) {
                for (const auto& socket : sockets) {
                    socket->discard_pending_pages();
                }
            }
            done_word = word;
            if (!awaiting_acks && on_state && done_word == kernel_profiler::kResidentAwaitingAcksWord) {
                report(RelayState::AwaitingAcks);
                awaiting_acks = true;
            }
            return done_word == kernel_profiler::kResidentDoneWord;
        });
    TT_FATAL(
        done,
        "streaming profiler: device {} {} did not finish within {} of its stop (done word {:#x})",
        ctx.chip_id,
        resident.name,
        kResidentTimeout,
        done_word);
    // The relay writes done after its socket barriers, so every byte has been acked by then.
    if (on_state) {
        report(RelayState::Done);
    }
    kernel_profiler::ResidentCtrl ctrl{};
    cluster.read_core(&ctrl, sizeof(ctrl), tt_cxy_pair(ctx.chip_id, resident.core.virt), resident.ctrl);
    if (ctrl.sync.dropped != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] device {} {} dropped {} of its {} records on a full ring",
            ctx.chip_id,
            resident.name,
            ctrl.sync.dropped,
            ctrl.sync.tail + ctrl.sync.dropped);
    }
}

void DevicePrograms::quiesce(const RelayStateFn& on_state) {
    if (quiesced_) {
        return;
    }
    quiesced_ = true;
    stop_links();
    // The check core stops before the wall-clock core, so every check reading has a clock point after it. The
    // wall-clock core stops before the eth relay, so the tail of its ring of clock sync records is final when the eth
    // relay drains it.
    enum Stage : size_t { Relays, Check, WallClock, EthRelay, StageCount };
    std::array<std::vector<std::pair<const DeviceCtx*, const ResidentCore*>>, StageCount> stages;
    for (const DeviceCtx& ctx : devices_) {
        const auto add = [&](Stage stage, const ResidentCore& resident) {
            if (resident.program != nullptr) {
                stages[stage].emplace_back(&ctx, &resident);
            }
        };
        for (const ResidentCore& relay : ctx.relays) {
            add(Relays, relay);
        }
        if (ctx.check) {
            add(Check, *ctx.check);
        }
        add(WallClock, ctx.wall_clock);
        add(EthRelay, ctx.eth_relay);
    }
    for (const auto& stage : stages) {
        for (const auto& [ctx, resident] : stage) {
            write_u32(
                mc_->get_cluster(),
                ctx->chip_id,
                resident->core.virt,
                resident->ctrl + offsetof(kernel_profiler::ResidentCtrl, stop),
                kernel_profiler::kResidentStopQuiesce);
        }
        for (const auto& [ctx, resident] : stage) {
            await_stop(*ctx, *resident, on_state);
        }
    }
    for (const DeviceCtx& ctx : devices_) {
        write_producers_armed(ctx, 0);
    }
}

// Reads each producer core's control vector once over MMIO, and reports its stall counts and any lane whose tail is
// still ahead of its head, since those words are missing from the capture.
void DevicePrograms::verify_completeness(uint32_t device_index) {
    const DeviceCtx& ctx = devices_[device_index];
    auto& cluster = mc_->get_cluster();
    std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
    uint64_t total = 0, stranded_words = 0, stranded_lanes = 0;
    std::vector<std::pair<uint32_t, uint32_t>> stalled;  // (stalls, core index)
    for (size_t ci = 0; ci < ctx.producers.size(); ci++) {
        cluster.read_core(
            cv.data(),
            kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, ctx.producers[ci].virt),
            ctx.producers[ci].control_vector_l1);
        uint32_t core_total = 0;
        for (uint32_t r = 0; r < kernel_profiler::SPSC_STALL_COUNT_MAX; r++) {
            core_total += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
        }
        total += core_total;
        if (core_total != 0) {
            stalled.emplace_back(core_total, static_cast<uint32_t>(ci));
        }
        for (uint32_t r = 0; r < kLanesPerTile; r++) {
            const int32_t left = static_cast<int32_t>(
                cv[kernel_profiler::SPSC_RING_TAIL_0 + r] - cv[kernel_profiler::SPSC_RING_HEAD_0 + r]);
            if (left > 0) {
                stranded_lanes++;
                stranded_words += static_cast<uint32_t>(left);
            }
        }
    }
    if (total != 0) {
        std::sort(stalled.begin(), stalled.end(), std::greater<>());
        std::string top;
        for (const auto& [count, ci] : stalled) {
            const CoreCoord& v = ctx.producers[ci].virt;
            top += fmt::format("{}({},{})#{}={}", top.empty() ? "" : " ", v.x, v.y, ci, count);
        }
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} profiler stalls on {} of {} cores; (virt x,y)#index=count: {}",
            ctx.chip_id,
            total,
            stalled.size(),
            ctx.producers.size(),
            top);
    }
    if (stranded_lanes != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} words on {} lanes were published after the relay's last sweep and are "
            "not in the capture",
            ctx.chip_id,
            stranded_words,
            stranded_lanes);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
