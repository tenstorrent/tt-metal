// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_profiler.hpp"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>  // MeshCoreCoord
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tools/profiler/x280_driver.hpp"
#include "tools/profiler/x280_profzone_boot.hpp"
#include "tools/profiler/x280_profzone_decode.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "impl/profiler/profiler.hpp"  // generateZoneSourceLocationsHashes (zone hash -> name)
#include "prof_packet.h"

namespace tt::tt_metal {

namespace pz = tt::tt_metal::profiler;

PerfDebugProfiler::DeviceCtx::DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::~DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

PerfDebugProfiler::PerfDebugProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    try {
        start(mesh_device);
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[perf-debug profiler] init failed ({}); disabled for this session.", e.what());
        stop();
    }
}

PerfDebugProfiler::~PerfDebugProfiler() { stop(); }

void PerfDebugProfiler::start(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[perf-debug profiler] not Blackhole; skipping X280 capture.");
        return;
    }

    tracy_ = std::make_unique<PerfDebugTracyHandler>();
    // NOTE: zone names are loaded LAZILY on the first drain (see drain_loop), NOT here -- at start()
    // (MeshDevice bring-up) the workload's kernels have not been JIT-compiled yet, so their zone-source-
    // location entries are not in the log and every name would fall back to "Zone_<hash>".

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.chip_id = static_cast<uint32_t>(mesh_device->get_device(coord)->id());
        if (!boot_device(mesh_device, ctx)) {
            continue;  // boot logs its own reason; degrade to no-capture for this device
        }
        // Tracy: anchor + pre-create the per-core contexts (off the drain hot path). Freq = device
        // aiclk in GHz (cycles/ns), matching the standard DeviceProfiler.
        double freq = cluster.get_device_aiclk(ctx.chip_id) / 1000.0;
        if (freq <= 0.0) {
            freq = 1.0;
        }
        tracy_->AddDevice(ctx.chip_id, tracy::Profiler::GetTime(), 0.0, freq);
        std::vector<std::pair<uint32_t, uint32_t>> worker_noc0;
        worker_noc0.reserve(ctx.core_virt.size());
        for (const auto& [vx, vy] : ctx.core_virt) {
            auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy);
            if (it != ctx.virt_to_noc0.end()) {
                worker_noc0.emplace_back(it->second.first, it->second.second);
            }
        }
        tracy_->PreCreateContexts(ctx.chip_id, worker_noc0);
        ctx.active = true;
        devices_.push_back(std::move(ctx));
    }

    // Spawn the continuous drain threads AFTER devices_ is stable (threads capture &devices_[i]).
    for (auto& ctx : devices_) {
        for (uint32_t s = 0; s < kNSockets; s++) {
            ctx.drain[s] = std::thread(&PerfDebugProfiler::drain_loop, this, std::ref(ctx), s);
        }
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] active on {} device(s): X280 drain (2 readers + 2 relays, 4 MiB sockets, "
            "adaptive) -> Tracy",
            devices_.size());
    }
}

bool PerfDebugProfiler::boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    if (soc.get_cores(CoreType::L2CPU, CoordSystem::NOC0).empty()) {
        return false;
    }
    std::string active_fw_path = BuildEnvManager::get_instance(context_id).get_x280_firmware_path(device_id);
    std::string idle_fw_path = BuildEnvManager::get_instance(context_id).get_x280_idle_firmware_path(device_id);
    if (active_fw_path.empty() || idle_fw_path.empty()) {
        return false;
    }
    auto read_file = [](const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        while (b.size() % 4 != 0) {
            b.push_back(0);
        }
        return b;
    };
    std::vector<uint8_t> active_fw = read_file(active_fw_path);
    std::vector<uint8_t> idle_fw = read_file(idle_fw_path);
    if (active_fw.empty() || idle_fw.empty()) {
        return false;
    }

    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t gx = static_cast<uint32_t>(grid.x), gy = static_cast<uint32_t>(grid.y);
    const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(num_cores) * kNRisc;
    ctx.core_virt.resize(num_cores);

    // MBOX_COORDS payload: per core {u32 virtual_x, u32 virtual_y} (what the X280 NoC-addresses), in
    // grid order (idx = ly*gx + lx) -- the SAME order the SRCLUT lane L=core*NRISC+risc resolves. Also
    // pre-zero each core's profiler control vector, and build the virtual->NOC0 map (for Tracy lanes).
    std::vector<uint8_t> coord_buf(num_cores * 8, 0);
    std::vector<uint8_t> zero_ctrl(256, 0);  // zero each core's profiler control vector (head/tail start clean)
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            const uint32_t idx = ly * gx + lx;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            const uint32_t vx = static_cast<uint32_t>(v.x), vy = static_cast<uint32_t>(v.y);
            std::memcpy(coord_buf.data() + idx * 8 + 0, &vx, 4);
            std::memcpy(coord_buf.data() + idx * 8 + 4, &vy, 4);
            cluster.write_core(zero_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, v), prof_l1);
            const CoreCoord noc0 = cluster.get_physical_coordinate_from_logical_coordinates(
                device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
            ctx.core_virt[idx] = {vx, vy};
            ctx.virt_to_noc0[(static_cast<uint64_t>(vx) << 32) | vy] = {
                static_cast<uint32_t>(noc0.x), static_cast<uint32_t>(noc0.y)};
        }
    }

    const auto pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    if (pcie_cores.empty()) {
        return false;
    }
    const auto pc = pcie_cores.front();
    const uint64_t pcie_enc = (static_cast<uint64_t>(pc.x) & 0x3f) | ((static_cast<uint64_t>(pc.y) & 0x3f) << 6);

    ctx.driver = std::make_unique<pz::X280Driver>(cluster, static_cast<int>(device_id), /*l2cpu=*/0);
    auto& drv = *ctx.driver;

    // Two D2HSockets (one per relay), sender = X280 L2CPU, config at the FW's X280_SOCKET_CONFIG_BASE
    // (0x08019000 + h*0x100). FIFO = 4 MiB (multi-window). Created BEFORE boot so the config md is
    // resident; the FIFO NoC addr is read back and packed into P_HOST_BASE (bytes_acked is host-written
    // live post-boot, so nothing needs to survive ensure_idle).
    const CoreCoord l2phys = pz::x280_l2cpu_tile(0);
    const distributed::MeshCoordinate scoord = *distributed::MeshCoordinateRange(mesh_device->shape()).begin();
    const uint32_t cfg_sz = distributed::D2HSocket::required_config_buffer_size();
    const uint64_t fifo_bytes = static_cast<uint64_t>(kHRingWords) * 4;
    uint64_t fifo_lo[kNSockets] = {0, 0};
    for (uint32_t h = 0; h < kNSockets; h++) {
        const uint32_t caddr = 0x08019000u + h * 0x100u;
        ctx.sockets[h] = std::make_unique<distributed::D2HSocket>(
            mesh_device,
            distributed::MeshCoreCoord{scoord, l2phys},
            static_cast<uint32_t>(fifo_bytes),
            distributed::D2HSocket::ExternalConfigBuffer{.address = caddr, .sender_is_l2cpu = true});
        ctx.sockets[h]->set_page_size(kPageSize);
        std::vector<uint8_t> cfgbuf(cfg_sz, 0);
        drv.read_block(cfgbuf.data(), cfg_sz, caddr);
        const uint32_t* c = reinterpret_cast<const uint32_t*>(cfgbuf.data());
        const uint64_t fifo = (static_cast<uint64_t>(c[13]) << 32) | c[4];
        fifo_lo[h] = fifo & 0xffffffffull;
        ctx.decode[h] = std::make_unique<pz::ProfzoneDecodeState>();
        ctx.decode[h]->reset(ctx.nl);
    }

    pz::ProfzoneBootCfg bcfg;
    bcfg.idle_fw = std::move(idle_fw);
    bcfg.active_fw = std::move(active_fw);
    bcfg.pll_mhz = 1000;
    bcfg.pcie_enc = pcie_enc;
    bcfg.host_base = static_cast<uint64_t>(fifo_lo[0]) | (static_cast<uint64_t>(fifo_lo[1]) << 32);
    bcfg.prof_l1 = prof_l1;
    bcfg.num_cores = num_cores;
    bcfg.hring_words = kHRingWords;
    bcfg.ndh = kNSockets;
    bcfg.nread = kNRead;
    bcfg.coords = coord_buf.data();
    bcfg.coords_bytes = static_cast<uint32_t>(coord_buf.size());
    bcfg.dualrelay = true;
    bcfg.adaptive = true;
    bcfg.socket = true;

    uint64_t nharts = 0;
    bool half_broken = false;
    if (!pz::boot_profzone(drv, bcfg, nharts, half_broken)) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: profzone bring-up failed (half_broken={}). If half_broken, "
            "`tt-smi -r {}` then rerun. Continuing without X280 capture.",
            device_id,
            half_broken,
            device_id);
        ctx.sockets[0].reset();
        ctx.sockets[1].reset();
        ctx.driver.reset();
        return false;
    }
    ctx.params_addr = pz::kProfzoneMboxParams;
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] Device {}: booted X280 drainer ({} cores, prof_l1=0x{:x}, pcie=({},{}))",
        device_id,
        num_cores,
        prof_l1,
        pc.x,
        pc.y);
    return true;
}

void PerfDebugProfiler::drain_loop(DeviceCtx& ctx, uint32_t sock_idx) {
    distributed::D2HSocket* sock = ctx.sockets[sock_idx].get();
    pz::ProfzoneDecodeState& st = *ctx.decode[sock_idx];
    const uint32_t page_words = kPageSize / sizeof(uint32_t);
    const uint32_t fifo_pages = sock->get_fifo_curr_size() / sock->get_page_size();
    std::vector<uint32_t> buf;
    auto backoff = std::chrono::microseconds(50);
    // Rebase device timestamps to the FIRST one this drain thread sees, so zones land near the Tracy
    // context origin (host_start) instead of ~device-wall-clock ticks into the timeline (a "multi-hour"
    // offset that renders zones off-screen). Matches test_x280_realprof / the RT handler's anchoring.
    uint64_t ts_base = 0;
    static const bool ddbg = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);
    uint64_t dbg_iters = 0, dbg_pages = 0, dbg_emit = 0;

    while (!stop_.load(std::memory_order_acquire)) {
        uint32_t np = sock->pages_available();
        if (np == 0) {
            std::this_thread::sleep_for(backoff);
            continue;
        }
        if (np >= fifo_pages) {
            np = fifo_pages - 1u;  // never read more than the FIFO holds (pages_available can spike)
        }
        if (ddbg && dbg_iters < 40) {
            log_info(tt::LogMetal, "[drain sock={}] iter={} np={} fifo_pages={}", sock_idx, dbg_iters, np, fifo_pages);
        }
        dbg_iters++;
        dbg_pages += np;
        buf.resize(static_cast<size_t>(np) * page_words);
        sock->read(buf.data(), np);  // auto-acks the sender

        // First drain with data => the workload's kernels have JIT-compiled, so the zone-source-location
        // log now holds their srcloc hashes. Load names ONCE (call_once blocks the sibling drain thread
        // until done, so the subsequent zone_names_ reads are race-free). Stable node storage => the
        // string_views handed to Tracy stay valid.
        std::call_once(names_once_, [this]() {
            try {
                for (auto& [h, md] : generateZoneSourceLocationsHashes()) {
                    zone_names_[h] = md.marker_name;
                }
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "[perf-debug profiler] zone-name load failed ({})", e.what());
            }
            zone_names_[0x7FFFu] = "X280-STALL";  // PROFILER_STALL_ZONE_ID
        });

        pz::profzone_decode(
            st,
            buf.data(),
            buf.size(),
            ctx.nl,
            [&](uint32_t lane, uint32_t type, uint32_t hash, uint64_t ts, uint32_t /*prog*/) {
                if (type != kernel_profiler::ZONE_START && type != kernel_profiler::ZONE_END) {
                    return;  // only START/END for now (DeviceZoneScopedN)
                }
                dbg_emit++;
                const uint32_t ci = lane / kNRisc, risc = lane % kNRisc;
                if (ci >= ctx.core_virt.size()) {
                    return;
                }
                // DIAG (TT_PERF_DEBUG_ZONE_DUMP=1): dump the first decoded markers' per-lane timestamp split
                // (hi = timer_hi, lo = timer_low) to spot a lane whose timer_hi never got set (-> zones land at
                // a wildly wrong time and "vanish" when zoomed to the good zones).
                static const bool zdump = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);
                static std::atomic<int> ndump{0};
                if (zdump && ndump.fetch_add(1, std::memory_order_relaxed) < 80) {
                    log_info(
                        tt::LogMetal,
                        "[zdump] ci={} risc={} hi={} lo={} ts={} start={} hash=0x{:x}",
                        ci,
                        risc,
                        (uint32_t)(ts >> 32),
                        (uint32_t)(ts & 0xffffffffu),
                        ts,
                        (type == kernel_profiler::ZONE_START),
                        hash);
                }
                const auto [vx, vy] = ctx.core_virt[ci];
                uint32_t nx = vx, ny = vy;
                if (auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy);
                    it != ctx.virt_to_noc0.end()) {
                    nx = it->second.first;
                    ny = it->second.second;
                }
                std::string_view name;
                if (auto it = zone_names_.find(static_cast<uint16_t>(hash)); it != zone_names_.end()) {
                    name = it->second;
                }
                perf_debug::WorkerZonePacket pkt;
                pkt.chip_id = ctx.chip_id;
                pkt.core_virtual_x = vx;
                pkt.core_virtual_y = vy;
                pkt.core_noc0_x = nx;
                pkt.core_noc0_y = ny;
                pkt.risc = risc;
                pkt.timer_id = hash;
                pkt.name = name;
                if (ts_base == 0) {
                    ts_base = ts;  // first device ts seen -> the rebase origin (maps to the context host_start)
                }
                pkt.timestamp = (ts >= ts_base) ? (ts - ts_base) : 0;
                pkt.is_start = (type == kernel_profiler::ZONE_START);
                tracy_->HandleWorkerZone(pkt);
            });
    }
    if (ddbg) {
        log_info(
            tt::LogMetal,
            "[drain sock={} EXIT] iters={} pages_read={} markers_emitted={}",
            sock_idx,
            dbg_iters,
            dbg_pages,
            dbg_emit);
    }
}

void PerfDebugProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }
    // Signal the X280 to end its drain (P_STOP) -- no reset; the idle FW stays resident.
    for (auto& ctx : devices_) {
        if (ctx.driver) {
            try {
                pz::profzone_stop(*ctx.driver);
            } catch (const std::exception&) {
            }
        }
    }
    stop_.store(true, std::memory_order_release);
    for (auto& ctx : devices_) {
        for (uint32_t s = 0; s < kNSockets; s++) {
            if (ctx.drain[s].joinable()) {
                ctx.drain[s].join();
            }
        }
    }
    tracy_.reset();
    devices_.clear();
}

}  // namespace tt::tt_metal
