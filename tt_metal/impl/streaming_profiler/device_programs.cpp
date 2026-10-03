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
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/sync/link_sync.hpp"
#include "impl/streaming_profiler/service.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

uint32_t packed_xy(const CoreCoord& core) {
    return kernel_profiler::word_of(
        kernel_profiler::NocXy{.x = static_cast<uint32_t>(core.x), .y = static_cast<uint32_t>(core.y)});
}

// Views 7 and 2 are the least reliable at bring-up, so they're dropped first.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
// Two generations of two slots, plus three the spool splits into its two bounce buffers.
constexpr uint32_t kStageSlots = 7;
// A core's record is 128 B, a power of two so the kernel indexes it with a shift. 72 covers two relays on a 140-core
// grid.
constexpr uint32_t kMaxRelayCores = 72;
constexpr uint32_t kScratchBytes = kMaxRelayCores * 128;
static_assert(kScratchBytes % 64 == 0);
constexpr uint32_t kPageSize = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint32_t kNRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
// Rides out multi-second page-fault stalls on the sync thread.
constexpr uint32_t kEthMinFifoBytes = 128u << 20;
constexpr uint32_t kEthSyncRingBytes = kernel_profiler::kSyncRingRecords * sizeof(kernel_profiler::SyncRecord);
// One 64 B block, which keeps the eth control block aligned for the eth relay's one 64 B NoC read of it.
constexpr uint32_t kCtrlBytes = 64;
static_assert(sizeof(kernel_profiler::ResidentCtrl) <= kCtrlBytes);
constexpr uint32_t kArmedOffset = kernel_profiler::PROFILER_ARMED * sizeof(uint32_t);
constexpr uint32_t kLinkCtl = offsetof(kernel_profiler::LinkSyncL1, ctl);
constexpr uint32_t kLinkDone = offsetof(kernel_profiler::LinkSyncL1, done);
constexpr auto kResidentTimeout = std::chrono::seconds(10);

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

void launch_compiled(IDevice* device, Program& program) {
    slow_dispatch::LaunchProgramAsync(*device, program, /*force_slow_dispatch=*/true);
}

}  // namespace

bool can_capture(const MetalContext& mc) {
    return mc.rtoptions().get_streaming_profiler_enabled() && mc.hal().get_arch() == tt::ARCH::BLACKHOLE &&
           mc.hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

CoreCoords locate(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type) {
    return CoreCoords{
        .logical = logical,
        .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
        .phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true)};
}

uint64_t host_l1_addr(const Hal& hal, HalProgrammableCoreType core, HalL1MemAddrType type) {
    return core == HalProgrammableCoreType::DRAM ? hal.get_dev_noc_addr(core, type) : hal.get_dev_addr(core, type);
}

void zero_l1(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t bytes) {
    const std::vector<uint8_t> zeros(bytes);
    cluster.write_core(zeros.data(), bytes, tt_cxy_pair(chip, virt), addr);
}

void zero_profiler_control(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr) {
    zero_l1(cluster, chip, virt, addr, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
}

uint8_t dram_view_endpoint_noc_mask(const metal_SocDescriptor& soc, const CoreCoord& logical) {
    return soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical));
}

void launch_resident(IDevice* device, Program& program) {
    compile_resident(device, program);
    launch_compiled(device, program);
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

tt::Cluster& DevicePrograms::get_cluster() const { return MetalContext::instance(context_id_).get_cluster(); }

void DevicePrograms::carve_l1(const Hal& hal) {
    // One D2H socket's config, in whole 64 B blocks so the eth control block below the configs stays 64 B aligned.
    const uint32_t cfg_bytes =
        tt::align(distributed::D2HSocket::required_config_buffer_size(hal.get_alignment(HalMemType::L1)), kCtrlBytes);
    const uint32_t slot_bytes = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    drisc_l1_.stage_base = base;
    drisc_l1_.core_records = drisc_l1_.stage_base + kStageSlots * slot_bytes;
    drisc_l1_.ctrl_local = drisc_l1_.core_records + kScratchBytes;
    drisc_l1_.cfg = base + region - cfg_bytes;
    drisc_l1_.ctrl = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) +
                     (drisc_l1_.ctrl_local - base);
    TT_FATAL(
        drisc_l1_.ctrl_local + kCtrlBytes <= drisc_l1_.cfg,
        "streaming profiler: DRISC L1 ({} B unreserved) cannot hold a relay's {} staging slots, core records and "
        "socket config",
        region,
        kStageSlots);

    const uint32_t eth_base = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t eth_size = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t need =
        2 * cfg_bytes + kCtrlBytes + slot_bytes + kernel_profiler::kEthSyncScratchBytes + kEthSyncRingBytes + kPageSize;
    eth_l1_.frames_cfg = eth_base + eth_size - 2 * cfg_bytes;
    eth_l1_.sync_cfg = eth_l1_.frames_cfg + cfg_bytes;
    eth_l1_.ctrl = eth_l1_.frames_cfg - kCtrlBytes;
    eth_l1_.stage = (eth_l1_.ctrl - slot_bytes) & ~(kPageSize - 1u);  // the pack pads assume a page-aligned slot
    eth_l1_.scratch = (eth_l1_.stage - kernel_profiler::kEthSyncScratchBytes) & ~(kPageSize - 1u);
    eth_l1_.sync_ring = eth_l1_.scratch - kEthSyncRingBytes;
    eth_l1_.link = link_sync::l1_addr(hal);
    eth_l1_.link_ring = eth_l1_.link + offsetof(kernel_profiler::LinkSyncL1, ring);
    TT_FATAL(
        eth_size >= need && eth_l1_.sync_ring >= eth_base,
        "streaming profiler: idle-eth L1 too small for the clock tracker ({} B unreserved, {} needed)",
        eth_size,
        need);

    const uint32_t bytes = MetalContext::instance(context_id_).rtoptions().get_streaming_profiler_spool_mb() << 20;
    if (bytes != 0) {
        drisc_l1_.spool_addr = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
        drisc_l1_.spool_bytes = bytes;
    }
}

bool DevicePrograms::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    context_id_ = mesh_device->impl().get_context_id();
    auto& mc = MetalContext::instance(context_id_);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const auto& rtopts = mc.rtoptions();
    if (!can_capture(mc)) {
        log_warning(
            tt::LogMetal, "[streaming profiler] not capturing: it needs Blackhole with DRAM programmable cores");
        return false;
    }
    fabric_link_sync_ = mc.get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
    capture_.sync_check = rtopts.get_streaming_profiler_sync_check_enabled();
    prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    carve_l1(hal);
    const uint32_t relay_fifo_bytes = rtopts.get_streaming_profiler_fifo_mb() << 20;
    const uint32_t eth_fifo_bytes = std::max(relay_fifo_bytes, kEthMinFifoBytes);

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx& ctx = devices_.emplace_back(*mesh_device, coord);
        ctx.chip_id = static_cast<uint32_t>(ctx.device->id());
        ctx.numa_node = static_cast<int>(cluster.get_numa_node_for_device(ctx.chip_id));
        ctx.capture.chip_id = ctx.chip_id;
        enumerate_worker_grid(mesh_device, ctx);
        enumerate_eth_cores(ctx);
        choose_relay_cores(mesh_device, ctx);

        const TileClocks* clocks = service().tile_clocks(context_id_, ctx.chip_id);
        TT_FATAL(clocks != nullptr, "streaming profiler: device {} has no tile clocks", ctx.chip_id);
        const int64_t tracker_offset = clocks->offset(CoreType::ETH, ctx.tracker.core.logical);
        for (size_t i = 0; i < ctx.producers.size(); i++) {
            const Producer& producer = ctx.producers[i];
            ctx.capture.tiles[i].clock_offset = tracker_offset - clocks->offset(producer.type, producer.logical);
        }
        if (ctx.ruler) {
            ctx.capture.ruler_offset = tracker_offset - clocks->offset(CoreType::ETH, ctx.ruler->core.logical);
        }
    }
    if (devices_.empty()) {
        return false;
    }
    const auto start_device = [&](DeviceCtx& ctx) {
        for (uint32_t d = 0; d < ctx.relays.size(); d++) {
            if (auto program = relay_program(ctx, d)) {
                start_resident(
                    ctx,
                    ctx.relays[d],
                    HalProgrammableCoreType::DRAM,
                    {{.cfg = drisc_l1_.cfg, .fifo_bytes = relay_fifo_bytes}},
                    std::move(program));
            }
        }
        start_resident(ctx, ctx.tracker, HalProgrammableCoreType::IDLE_ETH, {}, tracker_program(ctx));
        std::vector<SocketSpec> eth_sockets = {{.cfg = eth_l1_.sync_cfg, .fifo_bytes = eth_fifo_bytes, .sync = true}};
        if (rtopts.get_streaming_profiler_eth_enabled()) {
            eth_sockets.push_back({.cfg = eth_l1_.frames_cfg, .fifo_bytes = relay_fifo_bytes});
        }
        start_resident(ctx, ctx.eth_relay, HalProgrammableCoreType::IDLE_ETH, eth_sockets, eth_relay_program(ctx));
        write_producers_armed(ctx, 1);
    };
    std::vector<std::future<void>> starts;
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
    for (DeviceCtx& ctx : devices_) {
        capture_.devices.push_back(std::move(ctx.capture));
    }
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
    auto& cluster = get_cluster();
    std::vector<std::unique_ptr<distributed::D2HSocket>> made;
    for (const SocketSpec& spec : specs) {
        made.push_back(std::make_unique<distributed::D2HSocket>(
            ctx.mesh->shared_from_this(),
            distributed::MeshCoreCoord{ctx.coord, resident.core.phys},
            spec.fifo_bytes,
            distributed::D2HSocket::ExternalConfigBuffer{.address = spec.cfg, .sender_core_type = core_type},
            distributed::D2HSocket::ProcessScope::InProcess));
        made.back()->set_page_size(kPageSize);
    }
    // A previous capture's done, heartbeat and stop words would read as this one's.
    zero_l1(cluster, ctx.chip_id, resident.core.virt, resident.ctrl, kCtrlBytes);
    launch_resident(ctx.device, *program);
    // A relay launches fire-and-forget, so a core stuck in reset reports nothing. Two heartbeats prove its loop
    // runs.
    uint32_t heartbeat = 0;
    const bool started = poll_word(
        cluster,
        tt_cxy_pair(ctx.chip_id, resident.core.virt),
        resident.ctrl + offsetof(kernel_profiler::ResidentCtrl, heartbeat),
        std::chrono::milliseconds(500),
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
    resident.socket_index = static_cast<uint32_t>(ctx.sockets.size());
    resident.socket_count = static_cast<uint32_t>(made.size());
    for (size_t k = 0; k < made.size(); k++) {
        ctx.captured.push_back(CapturedSocket{
            .socket = made[k].get(),
            .dev = static_cast<uint32_t>(&ctx - devices_.data()),
            .index = static_cast<uint32_t>(ctx.sockets.size()),
            .numa_node = ctx.numa_node,
            .sync = specs[k].sync});
        ctx.sockets.push_back(std::move(made[k]));
    }
    resident.program = std::move(program);
}

DevicePrograms::Producer& DevicePrograms::add_producer(
    DeviceCtx& ctx, const CoreCoords& core, CoreType type, uint64_t prof_l1) {
    using experimental::streaming_profiler::Processor;
    // An eth core has two RISCs; its other lanes never carry a record.
    constexpr std::array<Processor, kNRisc> kTensix = {
        Processor::BRISC, Processor::NCRISC, Processor::TRISC0, Processor::TRISC1, Processor::TRISC2};
    constexpr std::array<Processor, kNRisc> kEth = {
        Processor::ERISC0, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1};
    ctx.capture.core_of_xy[packed_xy(core.virt)] = static_cast<uint16_t>(ctx.capture.tiles.size());
    ctx.capture.tiles.push_back({.xy = packed_xy(core.virt)});
    for (const Processor processor : type == CoreType::ETH ? kEth : kTensix) {
        ctx.capture.lanes.push_back(experimental::streaming_profiler::Core{
            .logical = core.logical,
            .physical = core.phys,
            .chip_id = static_cast<ChipId>(ctx.chip_id),
            .processor = processor});
    }
    return ctx.producers.emplace_back(Producer{core, type, prof_l1});
}

void DevicePrograms::enumerate_worker_grid(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = get_cluster();
    const uint32_t chip = ctx.chip_id;
    // A producer that isn't drained fills its ring and hangs the host at close.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            const Producer& producer = add_producer(
                ctx, locate(cluster, chip, CoreCoord{lx, ly}, CoreType::WORKER), CoreType::WORKER, prof_l1_);
            zero_profiler_control(cluster, chip, producer.virt, producer.prof_l1);
        }
    }
    ctx.worker_count = static_cast<uint32_t>(ctx.producers.size());

    // All Tensix cores: a dispatch core's ring is never drained and its L1 survives re-init, so a stale armed flag
    // there hangs close.
    const CoreCoord tensix_grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
    for (uint32_t ly = 0; ly < tensix_grid.y; ly++) {
        for (uint32_t lx = 0; lx < tensix_grid.x; lx++) {
            const CoreCoord virt =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord{lx, ly}, CoreType::WORKER);
            write_u32(cluster, chip, virt, prof_l1_ + kArmedOffset, 0);
        }
    }
}

void DevicePrograms::choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    const uint32_t chip = ctx.chip_id;
    const auto& soc = get_cluster().get_soc_desc(chip);
    const uint32_t view_count = static_cast<uint32_t>(soc.get_num_dram_views());
    ctx.relays.resize(std::min<uint32_t>(kRelayBankRoster.size(), view_count));
    std::vector<uint32_t> banks;
    std::ranges::copy_if(kRelayBankRoster, std::back_inserter(banks), [&](uint32_t bank) { return bank < view_count; });
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        ctx.relays[d].core.logical = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
    }
    // pick_unused_dram_logical_core() can't see two views resolving to one physical port, and two relays on one L1
    // would overlap.
    for (uint32_t a = 0; a < ctx.relays.size(); a++) {
        for (uint32_t b = a + 1; b < ctx.relays.size(); b++) {
            TT_FATAL(
                ctx.relays[a].core.logical != ctx.relays[b].core.logical,
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core.",
                a,
                banks[a],
                b,
                banks[b],
                ctx.relays[a].core.logical.x,
                ctx.relays[a].core.logical.y);
        }
    }
    // A channel carved into several views has an endpoint set per view, so one view's free subchannel can be another's
    // endpoint.
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        ResidentCore& relay = ctx.relays[d];
        const CoreCoord translated = soc.get_physical_dram_core_from_logical(relay.core.logical);
        const uint8_t noc2axi_mask = dram_view_endpoint_noc_mask(soc, relay.core.logical);
        TT_FATAL(
            noc2axi_mask == 0,
            "streaming profiler: device {} relay {}'s DRISC ({},{}) is a DRAM view's preferred endpoint on NOC mask "
            "{:#x}, so firmware holds those NIUs in NOC2AXI mode and the relay cannot initiate NoC traffic on them",
            chip,
            d,
            translated.x,
            translated.y,
            noc2axi_mask);
        const tt::umd::CoreCoord phys = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
        relay.core.phys = CoreCoord(phys.x, phys.y);
        relay.core.virt = ctx.device->virtual_core_from_logical_core(relay.core.logical, CoreType::DRAM);
        relay.name = fmt::format("relay {}", d);
        relay.ctrl = drisc_l1_.ctrl;
    }
}

std::unique_ptr<Program> DevicePrograms::relay_program(const DeviceCtx& ctx, uint32_t relay_index) {
    const ResidentCore& relay = ctx.relays[relay_index];
    const auto num_cores = static_cast<uint32_t>(ctx.workers().size());
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
        (num_cores + kMaxRelayCores - 1) / kMaxRelayCores,
        ctx.relays.size());
    if (my_cores == 0) {
        return nullptr;
    }
    // The relay is built with PROFILE_KERNEL and nothing drains its own profiler ring, which fills after about 74
    // launches in one reset window and hangs the RISC in firmware init.
    const auto& hal = MetalContext::instance(context_id_).hal();
    zero_profiler_control(
        get_cluster(),
        ctx.chip_id,
        relay.core.virt,
        host_l1_addr(hal, HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER));
    auto program = std::make_unique<Program>(CreateProgram());
    const KernelHandle kernel = CreateKernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/drisc_relay.cpp",
        relay.core.logical,
        // NOC 1 writes take about twice as long, so a relay there would take nearly every stall.
        DramConfig{
            .noc = NOC::NOC_0,
            .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}},
            .named_compile_args = {
                {"stage_base", drisc_l1_.stage_base},
                {"n_stage", kStageSlots},
                {"core_records", drisc_l1_.core_records},
                {"ctrl", drisc_l1_.ctrl_local},
                {"socket_config_addr", drisc_l1_.cfg},
                {"max_cores", kMaxRelayCores},
                // Spreads the relays over two request VCs.
                {"write_vc", (relay_index & 2u) ? 0u : 1u},
                {"spool_base", drisc_l1_.spool_addr},
                {"spool_bytes", drisc_l1_.spool_bytes}}});
    std::vector<uint32_t> runtime_args = {my_cores, static_cast<uint32_t>(prof_l1_)};
    // Reversed so the last-launched cores get the first-serviced slots.
    std::ranges::copy(
        std::span(ctx.capture.tiles).subspan(lo, my_cores) | std::views::reverse |
            std::views::transform(&CaptureContext::Device::Tile::xy),
        std::back_inserter(runtime_args));
    SetRuntimeArgs(*program, kernel, relay.core.logical, runtime_args);
    return program;
}

void DevicePrograms::enumerate_eth_cores(DeviceCtx& ctx) {
    auto& cluster = get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const uint32_t chip = ctx.chip_id;
    const std::vector<CoreCoord> idle = sorted_yx(ctx.device->get_inactive_ethernet_cores());
    TT_FATAL(
        idle.size() >= 2,
        "streaming profiler: device {} has {} idle ethernet cores; the clock tracker and its eth relay need two",
        chip,
        idle.size());
    std::vector<CoreCoords> idle_eth;
    std::ranges::transform(idle, std::back_inserter(idle_eth), [&](const CoreCoord& logical) {
        return locate(cluster, chip, logical, CoreType::ETH);
    });
    ctx.tracker = {.core = idle_eth.front(), .name = "clock tracker", .ctrl = eth_l1_.ctrl};
    const uint64_t idle_eth_prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::PROFILER);
    const Producer& tracker = add_producer(ctx, ctx.tracker.core, CoreType::ETH, idle_eth_prof_l1);
    zero_profiler_control(cluster, chip, tracker.virt, tracker.prof_l1);
    const auto nearest = [&](const CoreCoord& target) -> std::optional<CoreCoords> {
        auto others = idle_eth | std::views::drop(1) |
                      std::views::filter([&](const CoreCoords& other) { return other.phys != target; });
        const auto it = std::ranges::min_element(others, {}, [&](const CoreCoords& other) {
            return static_cast<uint32_t>(std::abs(static_cast<int>(other.phys.x) - static_cast<int>(target.x))) +
                   static_cast<uint32_t>(std::abs(static_cast<int>(other.phys.y) - static_cast<int>(target.y)));
        });
        return it == others.end() ? std::nullopt : std::optional(*it);
    };
    ctx.eth_relay = {.core = *nearest(ctx.tracker.core.phys), .name = "idle-eth relay", .ctrl = eth_l1_.ctrl};
    if (capture_.sync_check) {
        const std::optional<CoreCoords> ruler = nearest(ctx.eth_relay.core.phys);
        TT_FATAL(
            ruler.has_value(),
            "streaming profiler: device {} has {} idle ethernet cores; the sync check's ruler needs a third",
            chip,
            idle_eth.size());
        ctx.ruler = ResidentCore{.core = *ruler, .name = "sync check ruler", .ctrl = eth_l1_.ctrl};
    }
    const uint64_t active_eth_prof_l1 =
        hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
    for (const CoreCoord& logical :
         sorted_yx(ctx.device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true))) {
        const Producer& producer =
            add_producer(ctx, locate(cluster, chip, logical, CoreType::ETH), CoreType::ETH, active_eth_prof_l1);
        if (!fabric_link_sync_) {
            zero_profiler_control(cluster, chip, producer.virt, producer.prof_l1);
            continue;
        }
        // A fabric router already started this ring from its tail, so zeroing it would desync the tracker.
        std::array<uint32_t, kNRisc> tails{};
        cluster.read_core(
            tails.data(),
            sizeof(tails),
            tt_cxy_pair(chip, producer.virt),
            producer.prof_l1 + kernel_profiler::SPSC_RING_TAIL_0 * sizeof(uint32_t));
        cluster.write_core(
            tails.data(),
            sizeof(tails),
            tt_cxy_pair(chip, producer.virt),
            producer.prof_l1 + kernel_profiler::SPSC_RING_HEAD_0 * sizeof(uint32_t));
        // Safe to clear: nothing records before Run, and a core without a link end would keep an old tail.
        for (const uint32_t word : {kernel_profiler::SPSC_LINK_SYNC_TAIL, kernel_profiler::SPSC_LINK_SYNC_HEAD}) {
            write_u32(cluster, chip, producer.virt, producer.prof_l1 + word * sizeof(uint32_t), 0);
        }
    }
    for (const CoreCoords& idle_core : idle_eth | std::views::drop(1)) {
        if (idle_core.phys != ctx.eth_relay.core.phys && (!ctx.ruler || idle_core.phys != ctx.ruler->core.phys)) {
            const Producer& producer = add_producer(ctx, idle_core, CoreType::ETH, idle_eth_prof_l1);
            zero_profiler_control(cluster, chip, producer.virt, producer.prof_l1);
        }
    }
}

std::unique_ptr<Program> DevicePrograms::tracker_program(const DeviceCtx& ctx) {
    const CoreCoord& core = ctx.tracker.core.logical;
    zero_l1(
        get_cluster(),
        ctx.chip_id,
        ctx.tracker.core.virt,
        eth_l1_.scratch,
        offsetof(kernel_profiler::SyncSampleRing, samples));
    auto program = std::make_unique<Program>(CreateProgram());
    create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampler.cpp",
        core,
        DataMovementProcessor::RISCV_0,
        {{"ctrl_addr", eth_l1_.ctrl}, {"sample_ring_addr", eth_l1_.scratch}});
    create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_model.cpp",
        core,
        DataMovementProcessor::RISCV_1,
        {{"ctrl_addr", eth_l1_.ctrl}, {"sync_ring_addr", eth_l1_.sync_ring}, {"sample_ring_addr", eth_l1_.scratch}});
    return program;
}

std::unique_ptr<Program> DevicePrograms::eth_relay_program(const DeviceCtx& ctx) {
    auto program = std::make_unique<Program>(CreateProgram());
    const Producer& tracker = ctx.tracker_producer();
    std::unordered_map<std::string, uint32_t> relay_args = {
        {"frames_cfg", eth_l1_.frames_cfg},
        {"sync_cfg", eth_l1_.sync_cfg},
        {"stage", eth_l1_.stage},
        {"ctrl", eth_l1_.ctrl},
        {"scratch", eth_l1_.scratch},
        {"tracker_xy", packed_xy(tracker.virt)},
        {"tracker_prof_l1", static_cast<uint32_t>(tracker.prof_l1)},
        {"sync_ring", eth_l1_.sync_ring},
        {"link_ring", eth_l1_.link_ring}};
    if (ctx.ruler) {
        const CoreCoords& ruler = ctx.ruler->core;
        relay_args["ruler_xy"] = packed_xy(ruler.virt);
        zero_l1(get_cluster(), ctx.chip_id, ruler.virt, ctx.ruler->ctrl, kCtrlBytes);
        create_idle_eth_kernel(
            *program,
            "tt_metal/impl/streaming_profiler/kernels/eth_clock_ruler.cpp",
            ruler.logical,
            DataMovementProcessor::RISCV_0,
            {{"ctrl_addr", eth_l1_.ctrl}, {"sync_ring_addr", eth_l1_.sync_ring}});
    }
    const KernelHandle kernel = create_idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_relay.cpp",
        ctx.eth_relay.core.logical,
        DataMovementProcessor::RISCV_0,
        std::move(relay_args));
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
        runtime_args.push_back(static_cast<uint32_t>(producer.prof_l1));
    }
    SetRuntimeArgs(*program, kernel, ctx.eth_relay.core.logical, runtime_args);
    return program;
}

void DevicePrograms::write_producers_armed(const DeviceCtx& ctx, uint32_t armed) {
    auto& cluster = get_cluster();
    for (const Producer& producer : ctx.producers) {
        write_u32(cluster, ctx.chip_id, producer.virt, producer.prof_l1 + kArmedOffset, armed);
    }
}

void DevicePrograms::plan_links() {
    auto& mc = MetalContext::instance(context_id_);
    const auto core_of = [&](size_t dev, const CoreCoord& eth) {
        const DeviceCtx& ctx = devices_[dev];
        const auto drained = ctx.drained_by_eth_relay();
        const auto it = std::ranges::find(drained, eth, &Producer::logical);
        TT_FATAL(
            it != drained.end(),
            "streaming profiler: device {} link sync end eth({},{}) is not among the eth cores the eth relay drains",
            ctx.chip_id,
            eth.x,
            eth.y);
        return static_cast<uint32_t>(std::to_address(it) - ctx.producers.data());
    };
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            for (const link_sync::Link& link : link_sync::links_between(mc, chip_a, devices_[b].chip_id)) {
                const bool flip = link.chip_a != chip_a;
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
    std::vector<uint32_t> unreachable;
    for (size_t dev = 0; dev < devices_.size(); dev++) {
        if (!reached[dev]) {
            unreachable.push_back(devices_[dev].chip_id);
        }
    }
    TT_FATAL(
        unreachable.empty(),
        "streaming profiler: no link path to the root chip {} from chips {}",
        devices_[CaptureContext::kRootDevice].chip_id,
        fmt::format("{}", fmt::join(unreachable, ", ")));
}

std::array<DevicePrograms::LinkEnd, 2> DevicePrograms::link_ends(const CaptureContext::Link& link) const {
    const auto end = [&](uint32_t dev, uint32_t core, const CoreCoord& eth, bool sender) {
        const DeviceCtx& ctx = devices_[dev];
        const Producer& producer = ctx.producers[core];
        return LinkEnd{ctx.device, ctx.chip_id, eth, producer.virt, producer.prof_l1, sender};
    };
    return {end(link.dev_a, link.core_a, link.eth_a, true), end(link.dev_b, link.core_b, link.eth_b, false)};
}

void DevicePrograms::launch_links() {
    auto& cluster = get_cluster();
    links_running_ = true;
    if (!fabric_link_sync_) {
        for (const CaptureContext::Link& link : capture_.links) {
            const std::array ends = link_ends(link);
            for (const LinkEnd& end : ends) {
                zero_l1(cluster, end.chip, end.virt, eth_l1_.link, offsetof(kernel_profiler::LinkSyncL1, ring));
            }
            // Compile both first: the sender starts its handshake as soon as it runs.
            std::array<std::unique_ptr<Program>, 2> programs;
            for (size_t i = 0; i < ends.size(); i++) {
                programs[i] = std::make_unique<Program>(CreateProgram());
                const KernelHandle kernel = CreateKernel(
                    *programs[i],
                    "tt_metal/impl/streaming_profiler/kernels/link_sync.cpp",
                    ends[i].eth,
                    EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = {ends[i].sender ? 1u : 0u}});
                SetRuntimeArgs(*programs[i], kernel, ends[i].eth, {eth_l1_.link});
                compile_resident(ends[i].device, *programs[i]);
            }
            for (size_t i = 0; i < ends.size(); i++) {
                launch_compiled(ends[i].device, *programs[i]);
            }
            std::ranges::move(programs, std::back_inserter(resident_link_programs_));
        }
    }
    // A router's sender waits for Run like a resident one.
    for (const CaptureContext::Link& link : capture_.links) {
        const LinkEnd sender = link_ends(link)[0];
        write_u32(
            cluster,
            sender.chip,
            sender.virt,
            eth_l1_.link + kLinkCtl,
            static_cast<uint32_t>(kernel_profiler::LinkSyncCtl::Run));
    }
}

void DevicePrograms::stop_links() {
    if (!links_running_) {
        return;
    }
    auto& cluster = get_cluster();
    // Each end must record two solved rounds before any stops: the engine solves a link from two rounds, and an end
    // records a round when the next one starts. Under the sync check only every kLinkSyncCheckSolveEvery-th round is
    // solved.
    constexpr uint32_t kRecordsPerRound = 2;
    const uint32_t rounds_needed = capture_.sync_check ? kernel_profiler::kLinkSyncCheckSolveEvery + 1 : 2;
    for (const CaptureContext::Link& link : capture_.links) {
        for (const LinkEnd& end : link_ends(link)) {
            if (!poll_word(
                    cluster,
                    tt_cxy_pair(end.chip, end.virt),
                    end.prof_l1 + kernel_profiler::SPSC_LINK_SYNC_TAIL * sizeof(uint32_t),
                    std::chrono::seconds(1),
                    [&](uint32_t tail) { return tail >= kRecordsPerRound * rounds_needed; })) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] link sync chip {} {}: fewer than {} rounds recorded",
                    end.chip,
                    end.virt.str(),
                    rounds_needed);
            }
        }
    }
    // A router's receiver stops with its router; a resident end reports done.
    for (const CaptureContext::Link& link : capture_.links) {
        for (const LinkEnd& end : link_ends(link)) {
            if (fabric_link_sync_ && !end.sender) {
                continue;
            }
            write_u32(
                cluster,
                end.chip,
                end.virt,
                eth_l1_.link + kLinkCtl,
                static_cast<uint32_t>(kernel_profiler::LinkSyncCtl::Stop));
            if (!fabric_link_sync_) {
                const bool stopped = poll_word(
                    cluster,
                    tt_cxy_pair(end.chip, end.virt),
                    eth_l1_.link + kLinkDone,
                    kResidentTimeout,
                    [](uint32_t word) { return word != 0; });
                TT_FATAL(
                    stopped,
                    "streaming profiler: device {} link sync {} {} did not stop within {}",
                    end.chip,
                    end.sender ? "sender" : "receiver",
                    end.virt.str(),
                    kResidentTimeout);
            }
        }
    }
    resident_link_programs_.clear();
    links_running_ = false;
}

void DevicePrograms::start() {
    auto& cluster = get_cluster();
    const uint64_t go_addr = eth_l1_.ctrl + offsetof(kernel_profiler::ResidentCtrl, go);
    for (const DeviceCtx& ctx : devices_) {
        write_u32(cluster, ctx.chip_id, ctx.tracker.core.virt, go_addr, 1);
        write_u32(cluster, ctx.chip_id, ctx.eth_relay.core.virt, go_addr, 1);
    }
    // The ruler starts after the tracker's first instant, so its readings always have one before them.
    for (const DeviceCtx& ctx :
         devices_ | std::views::filter([](const DeviceCtx& device_ctx) { return device_ctx.ruler.has_value(); })) {
        const bool instant = poll_word(
            cluster,
            tt_cxy_pair(ctx.chip_id, ctx.tracker.core.virt),
            eth_l1_.ctrl + offsetof(kernel_profiler::ResidentCtrl, sync_tail),
            kResidentTimeout,
            [](uint32_t word) { return word != 0; });
        TT_FATAL(
            instant,
            "streaming profiler: device {} the clock tracker published no instant within {}",
            ctx.chip_id,
            kResidentTimeout);
        write_u32(cluster, ctx.chip_id, ctx.ruler->core.virt, go_addr, 1);
    }
    launch_links();
}

void DevicePrograms::request_stop(const DeviceCtx& ctx, const ResidentCore& resident) {
    write_u32(
        get_cluster(),
        ctx.chip_id,
        resident.core.virt,
        resident.ctrl + offsetof(kernel_profiler::ResidentCtrl, stop),
        kernel_profiler::kResidentStopQuiesce);
}

void DevicePrograms::await_stop(
    uint32_t device_index, const DeviceCtx& ctx, const ResidentCore& resident, const RelayStateFn& on_state) {
    const auto sockets = std::span(ctx.sockets).subspan(resident.socket_index, resident.socket_count);
    const auto report = [&](RelayState state) {
        for (uint32_t k = 0; k < resident.socket_count; k++) {
            on_state(device_index, resident.socket_index + k, state);
        }
    };
    auto& cluster = get_cluster();
    bool awaiting_acks = false;
    uint32_t state = 0;
    const bool done = poll_word(
        cluster,
        tt_cxy_pair(ctx.chip_id, resident.core.virt),
        resident.ctrl + offsetof(kernel_profiler::ResidentCtrl, done),
        kResidentTimeout,
        [&](uint32_t word) {
            // With no consumer, discard the pages so the relay's barriers still complete.
            for (const auto& socket : sockets) {
                if (!on_state && socket->pages_available() != 0) {
                    socket->discard_pending_pages();
                }
            }
            state = word;
            if (!awaiting_acks && on_state && state == kernel_profiler::kResidentAwaitingAcksWord) {
                report(RelayState::AwaitingAcks);
                awaiting_acks = true;
            }
            return state == kernel_profiler::kResidentDoneWord;
        });
    TT_FATAL(
        done,
        "streaming profiler: device {} {} did not finish within {} of its stop (state {:#x})",
        ctx.chip_id,
        resident.name,
        kResidentTimeout,
        state);
    // Done comes after the relay's socket barriers, so every byte is acked by then.
    if (on_state) {
        report(RelayState::Done);
    }
}

std::vector<const DevicePrograms::ResidentCore*> DevicePrograms::running_cores(const DeviceCtx& ctx, StopStage stage) {
    std::vector<const ResidentCore*> cores;
    const auto add_running = [&](const ResidentCore& core) {
        if (core.program) {
            cores.push_back(&core);
        }
    };
    switch (stage) {
        case StopStage::Relays:
            for (const ResidentCore& relay : ctx.relays) {
                add_running(relay);
            }
            return cores;
        case StopStage::Ruler:
            if (ctx.ruler_running()) {
                cores.push_back(&*ctx.ruler);
            }
            return cores;
        case StopStage::Tracker: add_running(ctx.tracker); return cores;
        case StopStage::EthRelay: add_running(ctx.eth_relay); return cores;
    }
    TT_THROW("Unreachable");
}

void DevicePrograms::quiesce(const RelayStateFn& on_state) {
    if (quiesced_) {
        return;
    }
    quiesced_ = true;
    stop_links();
    auto& cluster = get_cluster();
    // The ruler stops before the tracker, so its readings always have an instant after them, and the tracker before the
    // eth relay, so its ring's tail is final when the relay drains it.
    for (const StopStage stage : {StopStage::Relays, StopStage::Ruler, StopStage::Tracker, StopStage::EthRelay}) {
        // Every device is asked to stop a stage's cores before any is waited on, so they stop together.
        for (const DeviceCtx& ctx : devices_) {
            for (const ResidentCore* resident : running_cores(ctx, stage)) {
                request_stop(ctx, *resident);
            }
        }
        for (uint32_t device_index = 0; device_index < devices_.size(); device_index++) {
            const DeviceCtx& ctx = devices_[device_index];
            for (const ResidentCore* resident : running_cores(ctx, stage)) {
                await_stop(device_index, ctx, *resident, on_state);
            }
        }
    }
    for (const DeviceCtx& ctx : devices_) {
        for (const StopStage stage : {StopStage::Ruler, StopStage::Tracker}) {
            for (const ResidentCore* resident : running_cores(ctx, stage)) {
                kernel_profiler::ResidentCtrl ctrl{};
                cluster.read_core(&ctrl, sizeof(ctrl), tt_cxy_pair(ctx.chip_id, resident->core.virt), resident->ctrl);
                if (ctrl.dropped_sync != 0) {
                    log_warning(
                        tt::LogMetal,
                        "[streaming profiler] Device {}: {} dropped {} of its {} records on a full ring",
                        ctx.chip_id,
                        resident->name,
                        ctrl.dropped_sync,
                        ctrl.sync_tail + ctrl.dropped_sync);
                }
            }
        }
        write_producers_armed(ctx, 0);
    }
}

void DevicePrograms::verify_completeness() const {
    auto& cluster = get_cluster();
    for (const DeviceCtx& ctx : devices_) {
        std::array<uint32_t, kernel_profiler::SPSC_CONTROL_END> control{};
        uint64_t stranded_words = 0, stranded_lanes = 0;
        std::vector<std::pair<uint32_t, uint32_t>> stalled;
        for (size_t core_index = 0; core_index < ctx.producers.size(); core_index++) {
            const Producer& producer = ctx.producers[core_index];
            cluster.read_core(
                control.data(), sizeof(control), tt_cxy_pair(ctx.chip_id, producer.virt), producer.prof_l1);
            const auto stalls =
                std::span(control).subspan(kernel_profiler::SPSC_STALL_COUNT_0, kernel_profiler::SPSC_STALL_COUNT_MAX);
            if (const uint32_t total = std::accumulate(stalls.begin(), stalls.end(), 0u); total != 0) {
                stalled.emplace_back(total, static_cast<uint32_t>(core_index));
            }
            for (uint32_t r = 0; r < kNRisc; r++) {
                const int32_t left = static_cast<int32_t>(
                    control[kernel_profiler::SPSC_RING_TAIL_0 + r] - control[kernel_profiler::SPSC_RING_HEAD_0 + r]);
                if (left > 0) {
                    stranded_lanes++;
                    stranded_words += static_cast<uint32_t>(left);
                }
            }
        }
        if (!stalled.empty()) {
            std::ranges::sort(stalled, std::greater<>());
            uint64_t total = 0;
            std::string cores;
            for (const auto& [count, core_index] : stalled) {
                const CoreCoord& virt = ctx.producers[core_index].virt;
                cores += fmt::format("{}({},{})#{}={}", cores.empty() ? "" : " ", virt.x, virt.y, core_index, count);
                total += count;
            }
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} profiler stalls on {} of {} cores; (virt x,y)#index=count: {}",
                ctx.chip_id,
                total,
                stalled.size(),
                ctx.producers.size(),
                cores);
        }
        if (stranded_lanes != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} words on {} lanes were published after their relay's last sweep "
                "and are not in the capture",
                ctx.chip_id,
                stranded_words,
                stranded_lanes);
        }
    }
}

}  // namespace tt::tt_metal::streaming_profiler
