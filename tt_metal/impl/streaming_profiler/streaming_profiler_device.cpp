// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_device.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>  // MeshCoreCoord
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/tlb.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig (a DRISC kernel is not in the public headers yet)
#include "llrt/tt_cluster.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"
#include "impl/streaming_profiler/streaming_profiler_tile_clocks.hpp"
#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Views 7 and 2 last: a roster truncated to fewer relays sheds the bring-up-fragile views first.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
// Staging slots per relay, capped at what a DRISC's L1 fits; the spool's bounce slots need all of them, direct push
// the relay's two generations of two slots.
constexpr uint32_t kMaxStageSlots = 7;
constexpr uint32_t kMinStageSlots = 4;
// One 128-byte record per core (control-vector words 12..31, head mirror, wire XY; a power of two keeps the kernel's
// record addressing a shift) and byte-indexed core lists in the kernel. 72 covers two relays on a 140-core grid; a
// relay handed more cores refuses the capture with a message.
constexpr uint32_t kMaxRelayCores = 72;
constexpr uint32_t kScratchBytes = kMaxRelayCores * 128;
static_assert(kScratchBytes % 64 == 0);
constexpr uint32_t kCfgReserve = 8 * 1024;
constexpr uint32_t kMiscBytes = 1024;  // done(64) + stop(64), with headroom
constexpr uint32_t kPageSize = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint32_t kNRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
// Idle-eth drainers: two sockets per drainer, the linked cores' profiler frames and the sync's records. The routers'
// own zones are well under 1 MB/s, so 1 MiB of host FIFO (a single 2 MiB-aligned carve of the host channel) is
// generous for the frames. The sync stream runs ~3.6 MB/s (a drainer anchor every poll, the pusher's instants),
// bursting to ~70 MB/s through a glide. A record the sync thread has not read when the device laps it is lost, and
// the clock model bridges the lost instants with a line; that thread has been seen blocked for 3.3 s in a single page
// fault while the host process freed memory, so the sync FIFO holds half a minute of the steady stream.
constexpr uint32_t kEthFifoBytes = 1u << 20;
constexpr uint32_t kEthSyncFifoBytes = 128u << 20;
constexpr uint32_t kEthPllBytes = 64;  // the pusher's PLL reads land at the register's offset in an aligned 64 B
constexpr uint32_t kEthSyncRingBytes = kernel_profiler::kSyncRingBytes;
// A drainer's control block: done and heartbeat words, then the stop word one stride up.
constexpr uint32_t kCtrlBytes = 2 * kernel_profiler::kRelayCtrlWordStride;
// Drainer scratch (eth_clock_drainer.cpp): a linked core's control vector and two ring images (BH eth has DM0 and
// DM1), the sync record images, and the anchor audit.
constexpr uint32_t kEthScratchBytes = 16384;
static_assert(
    kEthScratchBytes >= kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + 2 * kernel_profiler::PROFILER_L1_BUFFER_SIZE,
    "the pusher scratch must hold a control vector and two whole rings");
// The worker wall clock's rate, to a few parts in 1e5: for tick-to-time conversions in logs and bounds. The sync
// engine measures the clocks it places records by.
double measure_frequency_ghz(tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker) {
    constexpr uint64_t kWallClockL = 0xFFB121F0ULL;  // RISCV_DEBUG_REG_WALL_CLOCK_L; reading it latches H
    constexpr uint64_t kWallClockH = 0xFFB121F8ULL;
    const tt_cxy_pair target(chip_id, worker);
    const auto read = [&]() {
        uint32_t lo = 0, hi = 0;
        const auto t0 = std::chrono::steady_clock::now();
        cluster.read_reg(&lo, target, kWallClockL);
        cluster.read_reg(&hi, target, kWallClockH);
        const auto t1 = std::chrono::steady_clock::now();
        return std::pair{t0 + (t1 - t0) / 2, (static_cast<uint64_t>(hi) << 32) | lo};
    };
    const auto [h0, d0] = read();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto [h1, d1] = read();
    return static_cast<double>(d1 - d0) /
           static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(h1 - h0).count());
}

CoreCoords locate(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type) {
    return CoreCoords{
        .logical = logical,
        .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
        .phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true)};
}

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> out(cores.begin(), cores.end());
    std::sort(out.begin(), out.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });
    return out;
}

void write_u32(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t value) {
    cluster.write_core(&value, sizeof(value), tt_cxy_pair(chip, virt), addr);
}

// Reads the word at `addr` until `pred` accepts it; false once `timeout` passes.
template <class Pred>
bool poll_word(
    tt::Cluster& cluster, const tt_cxy_pair& core, uint64_t addr, std::chrono::milliseconds timeout, Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
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

// A resident drainer launches fire-and-forget, so a core that never leaves reset produces no error and the
// workload wedges on full rings; the heartbeat counts sweeps, and two of them prove the loop runs.
bool heartbeat_advanced(
    tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t hb_addr, std::string_view what) {
    uint32_t hb = 0;
    if (poll_word(cluster, tt_cxy_pair(chip, virt), hb_addr, std::chrono::milliseconds(500), [&](uint32_t w) {
            hb = w;
            return w >= 2;
        })) {
        return true;
    }
    log_warning(
        tt::LogMetal,
        "[streaming profiler] Device {}: {} FAILED TO START (heartbeat {} after launch); nothing drains from it",
        chip,
        what,
        hb);
    return false;
}

std::unique_ptr<distributed::D2HSocket> make_socket(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const distributed::MeshCoordinate& coord,
    const CoreCoord& sender_phys,
    uint32_t fifo_bytes,
    uint32_t cfg_addr,
    HalProgrammableCoreType sender_core_type) {
    auto socket = std::make_unique<distributed::D2HSocket>(
        mesh_device,
        distributed::MeshCoreCoord{coord, sender_phys},
        fifo_bytes,
        distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_addr, .sender_core_type = sender_core_type},
        distributed::D2HSocket::ProcessScope::InProcess);
    socket->set_page_size(kPageSize);
    return socket;
}

// Compiles, writes the runtime args and launches without waiting: the drainers and the sync's kernels are resident.
void launch_resident(IDevice* device, Program& program) {
    detail::CompileProgram(device, program, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(device, program, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
}

std::string_view first_line(std::string_view s) { return s.substr(0, s.find('\n')); }

}  // namespace

Devices::DeviceCtx::DeviceCtx() = default;
Devices::DeviceCtx::~DeviceCtx() = default;
Devices::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

Devices::Devices() = default;
Devices::~Devices() = default;

bool Devices::carve_relay_l1(const Hal& hal) {
    prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    drisc_l1_base_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    drisc_l1_noc_ = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    slot_bytes_ = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
    l1_.n_stage = std::min(region > fixed ? (region - fixed) / slot_bytes_ : 0u, kMaxStageSlots);
    if (l1_.n_stage < kMinStageSlots) {
        log_warning(tt::LogMetal, "[streaming profiler] DRISC L1 too small for a relay; skipping");
        return false;
    }
    l1_.stage_base = drisc_l1_base_;
    l1_.core_records = l1_.stage_base + l1_.n_stage * slot_bytes_;
    l1_.done = l1_.core_records + kScratchBytes;
    l1_.stop = l1_.done + kernel_profiler::kRelayCtrlWordStride;
    l1_.cfg = drisc_l1_base_ + region - kCfgReserve;
    TT_FATAL(l1_.stop + kernel_profiler::kRelayCtrlWordStride <= l1_.cfg, "DRISC L1 layout overlaps the socket config");
    return true;
}

// Carved from the top of IDLE_ETH UNRESERVED down: socket config, ctrl words, one frame slot (the same slot geometry
// as a relay, since the eth core is enumerated as a standard 5-lane core), the linked-core scratch, the tile table,
// the sample ring.
void Devices::carve_eth_l1(const Hal& hal, uint32_t& aeth_unreserved, uint32_t& aeth_unres_size) {
    eth_ok_ = false;
    aeth_ok_ = false;
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        try {
            aeth_prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
            aeth_unreserved = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
            aeth_unres_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
            aeth_ok_ = aeth_prof_l1_ != 0 && aeth_unres_size >= 64;
        } catch (const std::exception&) {
            aeth_ok_ = false;
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        eth_prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::PROFILER);
        const uint32_t ebase = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        const uint32_t esize = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        const uint32_t need =
            kCfgReserve + kCtrlBytes + slot_bytes_ + kEthScratchBytes + kEthPllBytes + kEthSyncRingBytes + kPageSize;
        if (esize >= need) {
            eth_l1_.cfg = ebase + esize - kCfgReserve;
            eth_l1_.sync_cfg = eth_l1_.cfg + kCfgReserve / 2;
            eth_l1_.ctrl = eth_l1_.cfg - kCtrlBytes;
            eth_l1_.stage =
                (eth_l1_.ctrl - slot_bytes_) & ~(kPageSize - 1u);  // the pack pads assume a page-aligned slot
            eth_l1_.scratch = (eth_l1_.stage - kEthScratchBytes) & ~(kPageSize - 1u);
            eth_l1_.pll = eth_l1_.scratch - kEthPllBytes;
            eth_l1_.sync_ring = eth_l1_.pll - kEthSyncRingBytes;
            eth_l1_.link_ring = aeth_ok_ ? aeth_unreserved + aeth_unres_size - kernel_profiler::kLinkSyncL1Bytes +
                                               kernel_profiler::kLinkSyncRingOffset
                                         : 0u;
            eth_ok_ = eth_l1_.sync_ring >= ebase;
        }
        if (!eth_ok_) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] idle-eth L1 too small for a clock pusher ({} B unreserved, {} needed); eth clock "
                "tracking is OFF",
                esize,
                need);
        }
    }
}

std::vector<CapturedDevice> Devices::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    context_id_ = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[streaming profiler] not Blackhole; skipping relay capture.");
        return {};
    }
    // The relay is a DRISC: one DM RISC-V on a DRAM core, which today exists only on Blackhole.
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] no DRAM programmable cores (card FW below the DRISC gate?); producers stay unarmed "
            "and markers are DROPPED");
        return {};
    }
    if (!carve_relay_l1(hal)) {
        return {};
    }
    uint32_t aeth_unreserved = 0, aeth_unres_size = 0;
    carve_eth_l1(hal, aeth_unreserved, aeth_unres_size);

    sync_ = std::make_unique<SyncDevices>(context_id_, eth_l1_, aeth_unreserved, aeth_unres_size);
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.device = mesh_device->get_device(coord);
        ctx.chip_id = static_cast<uint32_t>(ctx.device->id());
        bool up = false;
        try {
            up = boot_device(mesh_device, ctx, coord);
        } catch (const std::exception& e) {
            // Stopped while every socket still exists: a drainer's last push into memory this process no longer
            // maps is an IOMMU fault against the device.
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: bring-up failed ({}); disabled for this session.",
                ctx.chip_id,
                first_line(e.what()));
            devices_.push_back(std::move(ctx));
            try {
                quiesce({});
            } catch (const std::exception& stop_error) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] stopping the drainers after the failed bring-up also failed ({})",
                    first_line(stop_error.what()));
            }
            return {};
        }
        if (!up) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: no DRISC relay -- producers stay unarmed (markers are DROPPED, but "
                "the workload will not stall waiting for a consumer)",
                ctx.chip_id);
            stop_device(static_cast<uint32_t>(devices_.size()), ctx, {});
            sync_->truncate(static_cast<uint32_t>(devices_.size()));
            continue;
        }
        ctx.out.chip_id = ctx.chip_id;
        ctx.out.numa_node =
            static_cast<int>(MetalContext::instance(context_id_).get_cluster().get_numa_node_for_device(ctx.chip_id));
        devices_.push_back(std::move(ctx));
    }
    std::vector<CapturedDevice> out;
    for (DeviceCtx& ctx : devices_) {
        out.push_back(std::move(ctx.out));
    }
    if (!devices_.empty()) {
        sync_->plan_links();
        sync_->start_probe();
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] active on {} device(s){}",
            devices_.size(),
            rtopts.get_streaming_profiler_tracy_enabled() ? " with the Tracy sink" : "");
    }
    return out;
}

bool Devices::boot_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
    enumerate_worker_grid(mesh_device, ctx);
    if (eth_ok_) {
        enumerate_eth_cores(ctx);
    }
    if (!choose_relay_cores(mesh_device, ctx)) {
        return false;
    }
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    ctx.out.ctx.frequency_ghz = measure_frequency_ghz(cluster, ctx.chip_id, ctx.producers.front().virt);
    TT_FATAL(ctx.out.ctx.frequency_ghz > 0.0, "streaming profiler: device {} wall clock did not advance", ctx.chip_id);
    SyncDevices::Device sd{
        .chip_id = ctx.chip_id, .device = ctx.device, .eth = ctx.idle_eth, .frequency_ghz = ctx.out.ctx.frequency_ghz};
    sd.tensix.assign(ctx.producers.begin(), ctx.producers.begin() + ctx.n_workers);
    if (ctx.pusher) {
        sd.linked.assign(ctx.producers.begin() + ctx.n_workers + 1, ctx.producers.end());
    }
    if (ctx.eth_drainer) {
        sd.drainer = ctx.eth_drainer->core;
    }
    const uint32_t si = sync_->add_device(std::move(sd));
    ctx.out.ctx.tile_offset.assign(ctx.out.ctx.core_xy.size(), 0);
    if (ctx.pusher) {
        sync_->measure_tiles(si, ctx.out.ctx);
    }
    reserve_spool();
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        if (!launch_relay(mesh_device, ctx, coord, d)) {
            return false;
        }
    }
    ctx.out.n_relay_sockets = static_cast<uint32_t>(ctx.out.sockets.size());
    // The pusher comes up after the relays so its socket follows theirs (the receiver indexes sockets as a
    // contiguous prefix in launch order).
    if (ctx.pusher && !launch_eth_pusher(mesh_device, ctx, coord)) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: eth clock tracking is OFF for this device, the capture continues "
            "without it",
            ctx.chip_id);
        ctx.pusher.reset();
        ctx.eth_drainer.reset();
    }
    set_producers_armed(ctx, true);
    ctx.out.ctx.has_eth_tracker = ctx.pusher.has_value();
    return true;
}

Devices::Producer& Devices::enroll(DeviceCtx& ctx, const CoreCoords& core, uint64_t prof_l1, bool blocking) {
    CaptureContext::Device& cap = ctx.out.ctx;
    cap.core_xy.push_back(packed_xy(core.virt));
    for (uint32_t r = 0; r < kNRisc; r++) {
        cap.lanes.push_back(experimental::streaming_profiler::Core{
            .logical = core.logical,
            .physical = core.phys,
            .chip_id = static_cast<ChipId>(ctx.chip_id),
            .risc = static_cast<experimental::streaming_profiler::Risc>(r)});
    }
    ctx.producers.push_back(Producer{core, prof_l1, blocking});
    return ctx.producers.back();
}

void Devices::zero_control(const DeviceCtx& ctx, const Producer& p) {
    const std::vector<uint8_t> zero(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    MetalContext::instance(context_id_)
        .get_cluster()
        .write_core(zero.data(), static_cast<uint32_t>(zero.size()), tt_cxy_pair(ctx.chip_id, p.virt), p.prof_l1);
}

// Core identity is not in the packets: the relay stamps each frame with the NoC coordinate the host seeded it with,
// and core_xy resolves it to the core index the lanes are numbered by.
void Devices::enumerate_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const uint32_t chip = ctx.chip_id;
    // The poll list defines the drained set; a producer outside it fills its ring, blocks forever, and takes the
    // host down in wait_until_cores_done. The relay lives on a DRAM core, so the full grid is polled.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            zero_control(ctx, enroll(ctx, locate(cluster, chip, CoreCoord{lx, ly}, CoreType::WORKER), prof_l1_, true));
        }
    }
    ctx.n_workers = static_cast<uint32_t>(ctx.producers.size());

    // Producers boot unarmed. Upstream does this in BRISC firmware at boot; this tree keeps hw/firmware byte-for-byte
    // main, so the host does it here, before any relay launches, on every Tensix core of the device and not just the
    // compute grid zeroed above: a dispatch core's ring is drained by nobody and its L1 survives device re-init, so a
    // stale arm there would park its firmware zone in the stall path and wedge wait_until_cores_done() at close.
    // set_producers_armed() then sets the flag only on the cores the relays serve, once every relay is up.
    const CoreCoord tensix_grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
    for (uint32_t ly = 0; ly < tensix_grid.y; ly++) {
        for (uint32_t lx = 0; lx < tensix_grid.x; lx++) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord{lx, ly}, CoreType::WORKER);
            write_u32(cluster, chip, v, prof_l1_ + kernel_profiler::PROFILER_ARMED * sizeof(uint32_t), 0);
        }
    }
}

// Every relay's NIU into stream mode, in one launch, run to completion. D2HSocket construction writes its
// config into DRISC L1 from the host, which only lands once the NIU terminates inbound traffic at L1. One
// launch: every LaunchProgram carries a dram_barrier that MMIO-polls a core in every DRAM channel, and a
// barrier that reaches a core already in stream mode never completes.
bool Devices::choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const uint32_t chip = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(chip);

    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    if (nbanks == 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM views to host a relay -- the streaming profiler is OFF "
            "for this device.",
            chip);
        return false;
    }

    // One relay per DRAM view, up to the relay cap.
    ctx.relays.resize(std::min<uint32_t>(kMaxRelays, nbanks));

    std::vector<uint32_t> banks;
    for (const uint32_t b : kRelayBankRoster) {
        if (b < nbanks) {
            banks.push_back(b);
        }
    }
    TT_FATAL(
        banks.size() >= ctx.relays.size(),
        "streaming profiler needs {} relay banks but only {} usable DRAM views are in the roster (part has {} "
        "views)",
        ctx.relays.size(),
        banks.size(),
        nbanks);

    std::vector<CoreCoord> relay_cores;
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        ctx.relays[d].core.logical = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
        relay_cores.push_back(ctx.relays[d].core.logical);
    }
    // pick_unused_dram_logical_core() reserves per view and cannot see two views resolving to one physical port
    // (views 0 and 7 have both come back as NoC core 0-0); two relays on one L1 would silently overlap, so refuse.
    for (uint32_t a = 0; a < ctx.relays.size(); a++) {
        for (uint32_t b = a + 1; b < ctx.relays.size(); b++) {
            TT_FATAL(
                relay_cores[a] != relay_cores[b],
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core.",
                a,
                banks[a],
                b,
                banks[b],
                relay_cores[a].x,
                relay_cores[a].y);
        }
    }
    // A DRISC initiates NoC traffic only on an NIU firmware left in stream mode, and the relay uses both:
    // NOC_INDEX for egress, the other NoC for gathers. pick_unused_dram_logical_core() skips the endpoints of
    // the view it was asked for, but a channel carved into several views has one endpoint set per view, so the
    // free subchannel of one view can still be another's -- and firmware keeps that NIU in NOC2AXI, where the
    // relay's reads would never issue. Capture off rather than a relay whose gathers go nowhere.
    for (uint32_t d = 0; d < relay_cores.size(); d++) {
        const CoreCoord translated = soc.get_physical_dram_core_from_logical(relay_cores[d]);
        const uint8_t noc2axi_mask = soc.get_dram_endpoint_noc_mask(translated);
        if (noc2axi_mask != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: relay {}'s DRISC ({},{}) is a DRAM view's preferred endpoint on "
                "NOC mask {:#x}, so firmware holds those NIUs in NOC2AXI mode and the relay cannot initiate NoC "
                "traffic on them -- the streaming profiler is OFF for this device.",
                chip,
                d,
                translated.x,
                translated.y,
                noc2axi_mask);
            return false;
        }
    }
    return true;
}

void Devices::reserve_spool() {
    const auto& ctx = MetalContext::instance(context_id_);
    const uint32_t bytes = ctx.rtoptions().get_streaming_profiler_spool_mb() << 20;
    if (bytes == 0 || spool_bytes_ != 0) {
        return;
    }
    const auto& hal = ctx.hal();
    TT_FATAL(
        hal.get_dev_size(HalDramMemAddrType::PROFILER) >= bytes,
        "streaming profiler: the HAL's profiler DRAM region ({} B) is smaller than the {} B GDDR spool",
        hal.get_dev_size(HalDramMemAddrType::PROFILER),
        bytes);
    spool_addr_ = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
    spool_bytes_ = bytes;
}

// The relay kernel's geometry: its L1 layout, its socket, its share of the unicast VCs and the spool.
std::unordered_map<std::string, uint32_t> Devices::relay_compile_args(uint32_t chip, uint32_t d) const {
    const std::unordered_map<std::string, uint32_t> cargs = {
        {"stage_base", l1_.stage_base},
        {"n_stage", l1_.n_stage},
        {"core_records", l1_.core_records},
        {"done_addr", l1_.done},
        {"stop_addr", l1_.stop},
        {"socket_config_addr", l1_.cfg},
        {"max_cores", kMaxRelayCores},
        // d&2 splits the pushers across two of the four unicast request VCs.
        {"write_vc", (d & 2u) ? 0u : 1u},
        // The bounce slots cost a staging generation, so a smaller L1 falls back to direct push rather than
        // tripping the kernel's geometry static_asserts.
        {"spool_base", spool_addr_},
        {"spool_bytes", l1_.n_stage >= kMaxStageSlots ? spool_bytes_ : 0u}};
    if (spool_bytes_ != 0 && l1_.n_stage < kMaxStageSlots) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: only {} staging slots fit, too few for the spool's bounce "
            "buffers; relay {} runs direct push",
            chip,
            l1_.n_stage,
            d);
    }
    return cargs;
}

bool Devices::launch_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord,
    uint32_t d) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();
    const uint32_t chip = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(chip);
    Drainer& relay = ctx.relays[d];

    // Contiguous bands in the host's grid order: a core belongs to exactly one relay, and the integer prefix
    // split assigns every core once.
    const uint32_t num_cores = ctx.n_workers;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * d) / ctx.relays.size());
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * (d + 1)) / ctx.relays.size());
    const uint32_t my_cores = hi - lo;
    if (my_cores > kMaxRelayCores) {
        log_error(
            tt::LogMetal,
            "[streaming profiler] Device {}: relay {} would own {} cores but a relay holds at most {}; this "
            "{}-core grid needs {} relays and the part has {}",
            chip,
            d,
            my_cores,
            kMaxRelayCores,
            num_cores,
            (num_cores + kMaxRelayCores - 1) / kMaxRelayCores,
            ctx.relays.size());
        return false;
    }
    if (my_cores == 0) {
        return true;
    }
    const CoreCoord translated = soc.dram_bank_endpoint_coords.at(relay.core.logical.x).at(relay.core.logical.y);
    const tt::umd::CoreCoord phys = soc.translate_coord_to(
        tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
    relay.core.phys = CoreCoord(phys.x, phys.y);
    relay.core.virt = ctx.device->virtual_core_from_logical_core(relay.core.logical, CoreType::DRAM);
    relay.state_addr = relay_noc_addr(l1_.done);
    relay.stop_addr = relay_noc_addr(l1_.stop);

    // Zero the relay core's own profiler ring: the relay is built with PROFILE_KERNEL, firmware writes zone
    // markers into this ring on every launch, nothing drains it, and the SPSC backend blocks on a full ring, so
    // after ~74 launches in one reset window the RISC wedges in firmware init.
    const std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    cluster.write_core(
        zero_ctrl.data(),
        static_cast<uint32_t>(zero_ctrl.size()),
        tt_cxy_pair(chip, relay.core.virt),
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER));

    auto program = std::make_unique<Program>(CreateProgram());
    const std::unordered_map<std::string, uint32_t> cargs = relay_compile_args(chip, d);
    auto relay_id = CreateKernel(
        *program,
        "tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp",
        relay.core.logical,
        // NOC 1 egress runs ~2x the service interval of NOC 0, so a relay parked there takes essentially every
        // profiler stall.
        DramConfig{
            .noc = NOC::NOC_0, .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}}, .named_compile_args = cargs});
    std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1_)};
    // Reversed: launch order follows global index, so the slice's last-launched cores land in the first-chunk
    // slots, which are serviced first.
    for (uint32_t ci = hi; ci-- > lo;) {
        rt.push_back(ctx.out.ctx.core_xy[ci]);
    }
    SetRuntimeArgs(*program, relay_id, relay.core.logical, rt);
    return launch_drainer(
        mesh_device,
        ctx,
        coord,
        relay,
        DrainerL1{
            .core_type = HalProgrammableCoreType::DRAM,
            .cfg = l1_.cfg,
            .fifo_bytes = (rtopts.get_streaming_profiler_fifo_mb() << 20) / kPageSize * kPageSize},
        std::move(program),
        fmt::format("relay {}", d));
}

// One idle ethernet core per chip joins the DECODE roster as a standard 5-lane core: its DM0 lane carries its own
// firmware markers, every other lane is always empty, and the decoder skips a lane whose extent is 0 exactly as it
// does an idle TRISC. It never joins the relay roster: the core pushes its own ring over its own socket. The lowest
// (y, x) idle core, so the choice is stable run to run (the set is unordered).
void Devices::enumerate_eth_cores(DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const uint32_t chip = ctx.chip_id;
    const std::vector<CoreCoord> idle = sorted_yx(ctx.device->get_inactive_ethernet_cores());
    if (idle.empty()) {
        log_warning(
            tt::LogMetal, "[streaming profiler] Device {}: no idle ethernet core; eth clock tracking is OFF", chip);
        return;
    }
    for (const CoreCoord& l : idle) {
        ctx.idle_eth.push_back(locate(cluster, chip, l, CoreType::ETH));
    }
    if (ctx.idle_eth.size() < 2) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: one idle ethernet core, none left to drain it; eth clock tracking is "
            "OFF",
            chip);
        return;
    }
    Drainer pusher;
    pusher.core = ctx.idle_eth.front();
    zero_control(ctx, enroll(ctx, pusher.core, eth_prof_l1_, true));
    // The drainer: the idle core nearest the pusher on the NoC (fewest hops for its reads of the pusher's L1).
    Drainer drainer;
    drainer.core = ctx.idle_eth[1];
    uint32_t best = std::numeric_limits<uint32_t>::max();
    for (size_t i = 1; i < ctx.idle_eth.size(); i++) {
        const CoreCoord& a = ctx.idle_eth[i].phys;
        const CoreCoord& b = pusher.core.phys;
        const uint32_t hops = static_cast<uint32_t>(std::abs(static_cast<int>(a.x) - static_cast<int>(b.x))) +
                              static_cast<uint32_t>(std::abs(static_cast<int>(a.y) - static_cast<int>(b.y)));
        if (hops < best) {
            best = hops;
            drainer.core = ctx.idle_eth[i];
        }
    }
    ctx.eth_drainer = std::move(drainer);
    // The chip's active eth cores join the decode roster the same way and become this pusher's linked set. Only
    // cores no dispatch tunnel reserved; lowest (y, x) first.
    uint32_t linked = 0;
    if (aeth_ok_) {
        const bool fabric_on =
            MetalContext::instance(context_id_).get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
        for (const CoreCoord& al :
             sorted_yx(ctx.device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true))) {
            const Producer& p = enroll(ctx, locate(cluster, chip, al, CoreType::ETH), aeth_prof_l1_, false);
            linked++;
            if (!fabric_on) {
                zero_control(ctx, p);
                continue;
            }
            // A fabric router already owns this core and seeded its ring cursor from the tail it found; zeroing
            // the words now would have the pusher ship from 0 while the producer writes on from there. Adopt
            // the ring where it stands: the pusher starts at the tail.
            std::array<uint32_t, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE / sizeof(uint32_t)> cv{};
            cluster.read_core(cv.data(), sizeof(cv), tt_cxy_pair(chip, p.virt), p.prof_l1);
            for (uint32_t r = 0; r < kNRisc; r++) {
                cv[kernel_profiler::SPSC_RING_HEAD_0 + r] = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
            }
            cluster.write_core(
                cv.data(),
                static_cast<uint32_t>(kNRisc * sizeof(uint32_t)),
                tt_cxy_pair(chip, p.virt),
                p.prof_l1 + kernel_profiler::SPSC_RING_HEAD_0 * sizeof(uint32_t));
        }
    }
    ctx.out.ctx.n_eth_cores = 1u + linked;
    ctx.pusher = std::move(pusher);
}

bool Devices::launch_eth_pusher(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
    Drainer& p = *ctx.pusher;
    p.state_addr = eth_l1_.ctrl;
    p.stop_addr = eth_l1_.ctrl + kernel_profiler::kRelayCtrlWordStride;
    const auto arc = MetalContext::instance(context_id_)
                         .get_cluster()
                         .get_driver()
                         ->get_soc_descriptor(ctx.chip_id)
                         .get_cores(CoreType::ARC, CoordSystem::TRANSLATED);
    TT_FATAL(!arc.empty(), "streaming profiler: device {} has no ARC tile in its descriptor", ctx.chip_id);
    auto program = std::make_unique<Program>(CreateProgram());
    create_pusher_kernel(*program, eth_l1_, p.core, CoreCoord(arc.front().x, arc.front().y));
    if (!launch_drainer(
            mesh_device,
            ctx,
            coord,
            p,
            DrainerL1{.core_type = HalProgrammableCoreType::IDLE_ETH, .cfg = 0, .sync_cfg = 0, .fifo_bytes = 0},
            std::move(program),
            "idle-eth pusher")) {
        return false;
    }
    // The drainer ships the pusher's own frames (its firmware markers), then every active eth core's; the pusher is
    // producers[n_workers], its linked cores follow.
    Drainer& dr = *ctx.eth_drainer;
    dr.state_addr = eth_l1_.ctrl;
    dr.stop_addr = eth_l1_.ctrl + kernel_profiler::kRelayCtrlWordStride;
    auto dprogram = std::make_unique<Program>(CreateProgram());
    const KernelHandle dkid = create_drainer_kernel(*dprogram, eth_l1_, dr.core, p.core);
    std::vector<uint32_t> rt = {static_cast<uint32_t>(ctx.producers.size() - ctx.n_workers)};
    for (size_t i = ctx.n_workers; i < ctx.producers.size(); i++) {
        rt.push_back(packed_xy(ctx.producers[i].virt));
        rt.push_back(static_cast<uint32_t>(ctx.producers[i].prof_l1));
    }
    rt.push_back(static_cast<uint32_t>(static_cast<int32_t>(ctx.out.ctx.drainer_offset)));
    SetRuntimeArgs(*dprogram, dkid, dr.core.logical, rt);
    if (!launch_drainer(
            mesh_device,
            ctx,
            coord,
            dr,
            DrainerL1{
                .core_type = HalProgrammableCoreType::IDLE_ETH,
                .cfg = eth_l1_.cfg,
                .sync_cfg = eth_l1_.sync_cfg,
                .fifo_bytes = kEthFifoBytes,
                .sync_fifo_bytes = kEthSyncFifoBytes},
            std::move(dprogram),
            "idle-eth drainer")) {
        return false;
    }
    ctx.out.sync_socket = dr.sock_idx + 1;
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: idle-eth clock pusher on eth ({},{}) up, its drainer on eth ({},{}) ships "
        "{} eth core(s)",
        ctx.chip_id,
        p.core.logical.x,
        p.core.logical.y,
        dr.core.logical.x,
        dr.core.logical.y,
        rt[0]);
    return true;
}

bool Devices::launch_drainer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord,
    Drainer& d,
    const DrainerL1& l1,
    std::unique_ptr<Program> program,
    std::string_view what) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    try {
        std::unique_ptr<distributed::D2HSocket> socket, sync_socket;
        if (l1.cfg != 0) {
            socket = make_socket(mesh_device, coord, d.core.phys, l1.fifo_bytes, l1.cfg, l1.core_type);
        }
        if (l1.sync_cfg != 0) {
            sync_socket = make_socket(mesh_device, coord, d.core.phys, l1.sync_fifo_bytes, l1.sync_cfg, l1.core_type);
        }
        // A stale done, heartbeat or stop word from the previous run reads as this run's live state (teardown leaves
        // stop at 1 or 2, and the drainer loop exits on nonzero stop).
        const std::array<uint32_t, kCtrlBytes / sizeof(uint32_t)> zero{};
        cluster.write_core(zero.data(), kCtrlBytes, tt_cxy_pair(ctx.chip_id, d.core.virt), d.state_addr);
        // A binary that failed to compile must NEVER reach LaunchProgram: launching onto an idle eth core holding no
        // valid binary once wedged the core and took the box down. CompileProgram throws into the catch below.
        launch_resident(ctx.device, *program);
        if (!heartbeat_advanced(cluster, ctx.chip_id, d.core.virt, d.state_addr + 4, what)) {
            return false;
        }
        d.sock_idx = static_cast<uint32_t>(ctx.out.sockets.size());
        d.n_sockets = 0;
        if (socket) {
            ctx.out.sockets.push_back(std::move(socket));
            d.n_sockets++;
        }
        if (sync_socket) {
            ctx.out.sockets.push_back(std::move(sync_socket));
            d.n_sockets++;
        }
        d.program = std::move(program);
    } catch (const std::exception& e) {
        // A code-region overflow fails the load, not the start, and the run then exits 0 with every marker dropped.
        log_error(tt::LogMetal, "[streaming profiler] Device {}: {} FAILED TO LOAD ({})", ctx.chip_id, what, e.what());
        return false;
    }
    return true;
}

// Producers boot unarmed (enumerate_worker_grid() clears PROFILER_ARMED on every Tensix core of the device) and
// only block on a full ring once armed, so a core no relay drains can never wedge device close. Arming follows the
// relays coming up; a relay that fails leaves the whole device unarmed and its markers are overwritten instead.
void Devices::set_producers_armed(const DeviceCtx& ctx, bool armed) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    for (const Producer& p : ctx.producers) {
        if (p.blocking) {
            write_u32(
                cluster,
                ctx.chip_id,
                p.virt,
                p.prof_l1 + kernel_profiler::PROFILER_ARMED * sizeof(uint32_t),
                armed ? 1u : 0u);
        }
    }
}

void Devices::release_eth_pushers() {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    for (const DeviceCtx& ctx : devices_) {
        if (ctx.pusher) {
            write_u32(cluster, ctx.chip_id, ctx.pusher->core.virt, eth_l1_.ctrl + 8, 1);
        }
        if (ctx.eth_drainer) {
            write_u32(cluster, ctx.chip_id, ctx.eth_drainer->core.virt, eth_l1_.ctrl + 8, 1);
        }
    }
}

void Devices::stop_drainer(
    uint32_t device_index,
    const DeviceCtx& ctx,
    const Drainer& r,
    std::string_view what,
    const RelayStateFn& on_state) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    // No consumer (a bring-up that failed): the drainer's barriers complete on the host's own acks.
    std::vector<distributed::D2HSocket*> socks;
    for (uint32_t k = 0; !on_state && k < r.n_sockets && r.sock_idx + k < ctx.out.sockets.size(); k++) {
        socks.push_back(ctx.out.sockets[r.sock_idx + k].get());
    }
    write_u32(cluster, ctx.chip_id, r.core.virt, r.stop_addr, kernel_profiler::kRelayStopQuiesce);
    bool drained = false;
    uint32_t state = 0;
    const bool done = poll_word(
        cluster, tt_cxy_pair(ctx.chip_id, r.core.virt), r.state_addr, std::chrono::seconds(10), [&](uint32_t w) {
            for (distributed::D2HSocket* sock : socks) {
                if (sock->pages_available() != 0) {
                    sock->discard_pending_pages();
                }
            }
            state = w & kernel_profiler::kRelayDoneMask;
            if (!drained && on_state && state == kernel_profiler::kRelayDrainedWord) {
                for (uint32_t k = 0; k < r.n_sockets; k++) {
                    on_state(device_index, r.sock_idx + k, RelayState::Drained);
                }
                drained = true;
            }
            return state == kernel_profiler::kRelayDoneWord;
        });
    TT_FATAL(
        done,
        "streaming profiler: device {} {} did not finish within 10 s of its stop (state {:#x})",
        ctx.chip_id,
        what,
        state);
    // Done follows the drainer's socket barriers, so the host has already acked every byte its sockets carry.
    for (uint32_t k = 0; on_state && k < r.n_sockets; k++) {
        on_state(device_index, r.sock_idx + k, RelayState::Done);
    }
}

void Devices::quiesce(const RelayStateFn& on_state) {
    if (sync_) {
        sync_->stop(MetalContext::instance(context_id_).get_cluster());
    }
    for (uint32_t di = 0; di < devices_.size(); di++) {
        stop_device(di, devices_[di], on_state);
    }
    // The relays have left their DRAM cores: those tiles read their rows again for the drift check.
    for (const DeviceCtx& ctx : devices_) {
        if (ctx.device == nullptr || ctx.relays.empty()) {
            continue;
        }
        try {
            for (const Drainer& r : ctx.relays) {
                if (r.program) {
                    detail::WaitProgramDone(ctx.device, *r.program, false);
                }
            }
            check_tile_clock_drift(ctx.device, context_id_);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: tile clock drift check skipped ({})",
                ctx.chip_id,
                first_line(e.what()));
        }
    }
}

void Devices::stop_device(uint32_t device_index, const DeviceCtx& ctx, const RelayStateFn& on_state) {
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        if (ctx.relays[d].program) {
            stop_drainer(device_index, ctx, ctx.relays[d], fmt::format("relay {}", d), on_state);
        }
    }
    // The pusher first: its ring's tail is final once it is done, and the drainer ships the rest before it stops.
    if (ctx.pusher && ctx.pusher->program) {
        stop_drainer(device_index, ctx, *ctx.pusher, "idle-eth pusher", on_state);
    }
    if (ctx.eth_drainer && ctx.eth_drainer->program) {
        stop_drainer(device_index, ctx, *ctx.eth_drainer, "idle-eth drainer", on_state);
    }
    // Nothing drains the rings any more: a producer blocked on a full one is released and overwrites from here on.
    set_producers_armed(ctx, false);
}

// One MMIO pass per roster core: the producer-owned stall counters, and each lane's tail against the consumed-words
// mirror.
void Devices::verify_completeness(uint32_t device_index) {
    const DeviceCtx& ctx = devices_[device_index];
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
    uint64_t total = 0, stranded_words = 0, stranded_lanes = 0;
    std::vector<std::pair<uint32_t, uint32_t>> stalled;  // (stalls, core index)
    for (size_t ci = 0; ci < ctx.producers.size(); ci++) {
        cluster.read_core(
            cv.data(),
            kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, ctx.producers[ci].virt),
            ctx.producers[ci].prof_l1);
        uint32_t core_total = 0;
        for (uint32_t r = 0; r < kernel_profiler::SPSC_STALL_COUNT_MAX; r++) {
            core_total += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
        }
        total += core_total;
        if (core_total != 0) {
            stalled.emplace_back(core_total, static_cast<uint32_t>(ci));
        }
        for (uint32_t r = 0; r < kNRisc; r++) {
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
            "[streaming profiler] Device {}: {} profiler stalls on {} of {} cores; (virt x,y)#index=stalls: {}",
            ctx.chip_id,
            total,
            stalled.size(),
            ctx.producers.size(),
            top);
    }
    if (stranded_lanes != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} words on {} lanes were published after their drainer's last sweep and "
            "are not in the capture",
            ctx.chip_id,
            stranded_words,
            stranded_lanes);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
