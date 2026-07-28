// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_profiler.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <common/TracyTTDeviceData.hpp>  // tracy::RiscType X280_RD0/X280_RELAY0 lanes

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

namespace {
// Read once: profile the X280 drain harts as well as the worker kernels.
bool hart_zones_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_HART_ZONES");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
}
}  // namespace

// Per-read page cap, overridable at runtime for tuning: TT_METAL_PERF_DEBUG_MAX_PAGES (0 = uncapped, take
// whatever the FIFO holds). The compiled default came from the synthetic benchmark; on high-volume real models
// (UFLD-v2: ~99M markers) the busier socket pins at the cap on every read, which is a suspect for the relay
// sitting in HOST-WAIT.
uint32_t max_pages_per_read(uint32_t compiled_default) {
    static const uint32_t v = [compiled_default] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_MAX_PAGES");
        if (s == nullptr || *s == '\0') {
            return compiled_default;
        }
        return static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

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
        // NOTE: per-core Tracy contexts are created LAZILY on each core's first zone (HandleWorkerZone ->
        // GetOrCreateContext). We deliberately do NOT pre-create the full worker grid here: only ~16 of
        // ~110 cores typically run the workload, and pre-creating all of them litters the capture with
        // empty (count=0) contexts that read as "cores not showing up". The per-zone mutex+lookup cost is
        // identical either way; lazy creation just avoids minting dead contexts.
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
            "[perf-debug profiler] active on {} device(s): X280 drain ({} readers + {} relays, {} MiB sockets, "
            "adaptive) -> Tracy",
            devices_.size(),
            kNRead,
            kNRelay,
            (static_cast<uint64_t>(kHRingWords) * 4) / (1024 * 1024));
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
    // TT_METAL_PERF_DEBUG_HART_ZONES=1: also profile the X280 DRAIN HARTS themselves (reader/relay busy +
    // stall spans), injected in-band. Enabling this also makes hart0 write the rdcycle->Tensix calibration
    // samples at boot, which stop() needs to place the hart spans on the device timeline. Off by default: it
    // adds ~24 B per drain-hart span to the same D2H stream the markers use.
    bcfg.hartzones = hart_zones_enabled();
    if (bcfg.hartzones) {
        ctx.nharts = kNRead + kNRelay;
        ctx.hz_raw.assign(ctx.nharts, {});
    }

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
    uint64_t dbg_iters = 0, dbg_pages = 0, dbg_emit = 0, dbg_stall = 0;

    // Drain-to-empty on stop: after stop_ is set (stop() sends the X280 P_STOP first, so it stops
    // producing), keep reading until the socket has been empty for a sustained window instead of exiting
    // on the first stop_ check -- otherwise the last in-flight markers (socket FIFO + host ring, ~the
    // pipeline depth of zones per lane) are abandoned and the tail of the run is lost from the capture.
    // A steady-clock deadline backstops the (shouldn't-happen) case where the socket never quiesces.
    uint32_t quiesce = 0;
    constexpr uint32_t kQuiesceEmpties = 200;  // ~10 ms sustained-empty at 50 us backoff => pipeline flushed
    std::chrono::steady_clock::time_point drain_deadline{};
    bool deadline_set = false;
    while (true) {
        const bool stopping = stop_.load(std::memory_order_acquire);
        if (stopping && !deadline_set) {
            drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            deadline_set = true;
        }
        uint32_t np = sock->pages_available();
        if (np == 0) {
            if (stopping && (++quiesce >= kQuiesceEmpties || std::chrono::steady_clock::now() >= drain_deadline)) {
                break;  // stop signalled AND socket drained (or deadline) => pipeline flushed, exit
            }
            std::this_thread::sleep_for(backoff);
            continue;
        }
        quiesce = 0;
        if (stopping && std::chrono::steady_clock::now() >= drain_deadline) {
            break;  // safety: socket still non-empty past the deadline (X280 not honoring P_STOP)
        }
        if (np >= fifo_pages) {
            np = fifo_pages - 1u;  // never read more than the FIFO holds (pages_available can spike)
        }
        const uint32_t cap = max_pages_per_read(kMaxPagesPerRead);
        if (cap != 0 && np > cap) {
            np = cap;  // bound one host turn; the loop takes the rest next iteration
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
                if (hash == 0x7FFFu && type == kernel_profiler::ZONE_START) {
                    dbg_stall++;  // PROFILER_STALL_ZONE_ID: a producer RISC blocked on a FULL ring, i.e. the
                                  // X280 drain did not keep up. Non-zero => the capture PERTURBS the workload
                                  // (kernels elongate by the stall); it is still lossless (the ring blocks
                                  // rather than dropping), but the timings are no longer clean.
                }
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
                    // Publish for push_hart_zones() so hart spans land on the SAME timeline as the kernels.
                    if (ctx.marker_ts_base == 0) {
                        ctx.marker_ts_base = ts;
                    }
                }
                pkt.timestamp = (ts >= ts_base) ? (ts - ts_base) : 0;
                pkt.is_start = (type == kernel_profiler::ZONE_START);
                tracy_->HandleWorkerZone(pkt);
            },
            // X280 drain-hart spans (only produced when bcfg.hartzones was set). Just accumulate here --
            // they cannot be placed on the timeline until stop(), which reads the rdcycle->Tensix calibration.
            // Each hart is written by exactly one drain thread, so this is race-free without a lock.
            [&](uint32_t hart, uint32_t meta, uint64_t rdc) {
                if (hart < ctx.hz_raw.size()) {
                    ctx.hz_raw[hart].push_back(DeviceCtx::HZMark{rdc, meta});
                }
            });
    }
    // Always report the per-socket drain totals: this is the only way to tell a healthy capture from a
    // silently-empty one. pages_read==0 means the X280 relayed nothing (bad boot / wrong grid);
    // markers_emitted is the device-zone count that actually reached Tracy.
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] socket {} drained: {} pages, {} markers -> Tracy ({} drain iterations); "
        "producer stall zones: {} [0 = X280 kept up, non-zero = capture perturbed the workload]",
        sock_idx,
        dbg_pages,
        dbg_emit,
        dbg_iters,
        dbg_stall);
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
    push_hart_zones();  // must run BEFORE tracy_.reset() -- it creates/uses Tracy contexts
    tracy_.reset();
    devices_.clear();
}

// Map the collected X280 drain-hart spans onto the device timeline and push them to Tracy as their own
// per-hart lanes ("rd0/rd1/relay0/relay1" in the GUI, via the widened RiscType). Runs once, at stop(), after
// the drain threads have joined so hz_raw is stable.
//
// The harts timestamp themselves with rdcycle (a fixed 1 GHz counter), NOT the Tensix wall clock the kernel
// markers use, so the two cannot be compared directly. hart0 therefore co-samples both clocks at boot (that
// is what bcfg.hartzones also switches on) and the host least-squares fits tensix = a*rdcycle + b here.
// Timestamps are rebased on the harts' OWN minimum, which cancels the constant offset between the
// calibration reference core's raw wall clock and the marker timeline (the same per-node origin trick the
// standalone harness uses; without it the whole lane can land before the origin and clamp to zero).
void PerfDebugProfiler::push_hart_zones() {
    if (!tracy_ || !hart_zones_enabled()) {
        return;
    }
    for (auto& ctx : devices_) {
        if (!ctx.driver || ctx.hz_raw.empty()) {
            continue;
        }
        uint64_t total = 0;
        for (const auto& v : ctx.hz_raw) {
            total += v.size();
        }
        if (total == 0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart zones enabled but none were captured.");
            continue;
        }
        // ---- rdcycle -> Tensix fit from hart0's boot co-samples {rdcycle_mid, tensix, noc_round_trip} ----
        const uint32_t nc = pz::kProfzoneCalibN;
        std::vector<uint64_t> raw(static_cast<size_t>(nc) * 3);
        try {
            ctx.driver->read_block(raw.data(), nc * 3 * sizeof(uint64_t), pz::kProfzoneCalibBase);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart-zone calib read failed ({})", e.what());
            continue;
        }
        std::vector<uint64_t> rts(nc);
        for (uint32_t i = 0; i < nc; i++) {
            rts[i] = raw[i * 3 + 2];
        }
        std::sort(rts.begin(), rts.end());
        const uint64_t rt_cut = rts[nc / 2] + rts[nc / 2] / 2;  // drop NoC-contended outliers
        const uint64_t x_base = raw[0], t_base = raw[1];
        double sx = 0, st = 0, sxx = 0, sxt = 0;
        uint32_t nfit = 0;
        for (uint32_t i = 0; i < nc; i++) {
            if (raw[i * 3 + 2] > rt_cut) {
                continue;
            }
            const double x = static_cast<double>(raw[i * 3 + 0] - x_base);
            const double t = static_cast<double>(raw[i * 3 + 1] - t_base);
            sx += x;
            st += t;
            sxx += x * x;
            sxt += x * t;
            nfit++;
        }
        if (nfit < 2 || (sxx * nfit - sx * sx) == 0.0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart-zone calib unusable (nfit={})", nfit);
            continue;
        }
        const double a = (sxt * nfit - sx * st) / (sxx * nfit - sx * sx);
        const double b = (st - a * sx) / nfit;
        auto map_ts = [&](uint64_t x) -> uint64_t {
            return static_cast<uint64_t>(a * static_cast<double>(x - x_base) + b) + t_base;
        };
        // Rebase on the MARKER origin so hart spans and kernel zones share one timeline. Falls back to the
        // harts' own minimum only if no markers were captured (nothing to align to).
        uint64_t hz_min = ~0ull, hz_max = 0;
        for (const auto& v : ctx.hz_raw) {
            for (const auto& m : v) {
                const uint64_t t = map_ts(m.rdc);
                hz_min = std::min(hz_min, t);
                hz_max = std::max(hz_max, t);
            }
        }
        const uint64_t marker_origin = ctx.marker_ts_base;
        const uint64_t hz_span_start = hz_min;  // kept for the diagnostic below
        if (marker_origin != 0) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart zones start {:.1f} ms BEFORE the first kernel marker (X280 drains "
                "from MeshDevice bring-up); aligning both on the marker origin",
                static_cast<double>(marker_origin - std::min(marker_origin, hz_min)) / 1.35e6);
            hz_min = marker_origin;
        }
        (void)hz_span_start;
        const CoreCoord l2t = pz::x280_l2cpu_tile(0);
        static constexpr uint32_t kHartColor[4] = {0xE67E22u, 0xF1C40Fu, 0x1ABC9Cu, 0x3498DBu};
        static constexpr uint32_t kBulkColor = 0xE74C3Cu;      // reader switched to BULK
        static constexpr uint32_t kHostWaitColor = 0x34495Eu;  // relay blocked on a full host FIFO
        static constexpr uint32_t kSpscWaitColor = 0x8E44ADu;  // reader blocked on a full LIM STAGE
        // ★ Push ALL harts as ONE CHRONOLOGICALLY SORTED stream, not hart-by-hart. Every hart lane lives in
        // the SAME Tracy GPU context, and Tracy's calibrated GPU-zone path expects timestamps to arrive in
        // increasing order per context. Pushing all of hart0, then all of hart1, ... makes the context's clock
        // jump backwards once per hart, which renders the lanes as separate blocks marched across the timeline
        // even though their real windows fully overlap (verified: identical 10-bucket time histograms, all
        // harts active throughout, min/max windows within 4 ms of each other).
        struct HZItem {
            uint64_t ts;
            uint32_t hart;
            uint32_t meta;
        };
        std::vector<HZItem> ordered;
        ordered.reserve(total);
        for (uint64_t h = 0; h < ctx.hz_raw.size(); h++) {
            for (const auto& m : ctx.hz_raw[h]) {
                ordered.push_back(HZItem{map_ts(m.rdc), static_cast<uint32_t>(h), m.meta});
            }
        }
        // stable_sort: a hart's own START/END pair share no timestamp ordering guarantee otherwise, and equal
        // timestamps must keep their emission order or a zone can close before it opens.
        std::stable_sort(
            ordered.begin(), ordered.end(), [](const HZItem& a, const HZItem& b) { return a.ts < b.ts; });
        std::vector<uint32_t> nz_per_hart(ctx.hz_raw.size(), 0);
        for (const auto& it : ordered) {
            const uint64_t h = it.hart;
            const bool is_reader = (h < kNRead);
            const std::string hname =
                is_reader ? ("X280 rd" + std::to_string(h)) : ("X280 relay" + std::to_string(h - kNRead));
            const uint32_t lane_risc = is_reader
                                           ? (static_cast<uint32_t>(tracy::RiscType::X280_RD0) + h)
                                           : (static_cast<uint32_t>(tracy::RiscType::X280_RELAY0) + (h - kNRead));
            const uint32_t is_start = it.meta & 1u;
            const uint32_t kind = (it.meta >> 1) & 3u;
            const char* suffix =
                (kind == 1) ? " BULK" : (kind == 2) ? " HOST-WAIT" : (kind == 3) ? " SPSC-WAIT" : "";
            perf_debug::WorkerZonePacket pkt;
            pkt.chip_id = ctx.chip_id;
            pkt.is_x280 = true;
            pkt.color = (kind == 1)   ? kBulkColor
                        : (kind == 2) ? kHostWaitColor
                        : (kind == 3) ? kSpscWaitColor
                                      : kHartColor[h & 3];
            pkt.core_noc0_x = static_cast<uint32_t>(l2t.x);
            pkt.core_noc0_y = static_cast<uint32_t>(l2t.y);
            pkt.risc = lane_risc;
            const std::string zn = hname + suffix;
            pkt.name = zn;
            pkt.timestamp = (it.ts >= hz_min) ? (it.ts - hz_min) : 0;
            pkt.is_start = (is_start != 0u);
            tracy_->HandleWorkerZone(pkt);
            nz_per_hart[h] += is_start;
        }
        for (uint64_t h = 0; h < nz_per_hart.size(); h++) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart {} ({}): {} zones -> Tracy (chronologically merged)",
                h,
                (h < kNRead) ? "READ" : "RELAY",
                nz_per_hart[h]);
        }
        for (uint64_t h = 0; h < 0; h++) {  // (old per-hart push loop retained below only for its diagnostics)
            const bool is_reader = (h < kNRead);
            const std::string hname =
                is_reader ? ("X280 rd" + std::to_string(h)) : ("X280 relay" + std::to_string(h - kNRead));
            const uint32_t lane_risc = is_reader
                                           ? (static_cast<uint32_t>(tracy::RiscType::X280_RD0) + h)
                                           : (static_cast<uint32_t>(tracy::RiscType::X280_RELAY0) + (h - kNRead));
            uint32_t nz = 0, n_end = 0;
            int depth_dbg = 0, max_depth_dbg = 0, unbalanced_dbg = 0;
            for (const auto& m : ctx.hz_raw[h]) {
                const uint32_t is_start = m.meta & 1u;
                const uint32_t kind = (m.meta >> 1) & 3u;  // 0=drain 1=bulk 2=hostwait 3=spscwait
                const char* suffix = (kind == 1) ? " BULK" : (kind == 2) ? " HOST-WAIT" : (kind == 3) ? " SPSC-WAIT" : "";
                const std::string zn = hname + suffix;
                const uint64_t ts = map_ts(m.rdc);
                perf_debug::WorkerZonePacket pkt;
                pkt.chip_id = ctx.chip_id;
                pkt.is_x280 = true;
                pkt.color = (kind == 1)   ? kBulkColor
                            : (kind == 2) ? kHostWaitColor
                            : (kind == 3) ? kSpscWaitColor
                                          : kHartColor[h & 3];
                pkt.core_noc0_x = static_cast<uint32_t>(l2t.x);
                pkt.core_noc0_y = static_cast<uint32_t>(l2t.y);
                pkt.risc = lane_risc;
                pkt.name = zn;
                pkt.timestamp = (ts >= hz_min) ? (ts - hz_min) : 0;
                pkt.is_start = (is_start != 0u);
                tracy_->HandleWorkerZone(pkt);
                nz += is_start;
                // DIAG: START/END balance. Unbalanced pairs make the Tracy handler nest ever deeper instead of
                // closing zones, which renders as giant cascading boxes. Under drain SATURATION the FW may not
                // have room to inject the closing marker, so this is the prime suspect there.
                if (is_start) {
                    depth_dbg++;
                    max_depth_dbg = std::max(max_depth_dbg, depth_dbg);
                } else {
                    n_end++;
                    if (depth_dbg == 0) {
                        unbalanced_dbg++;  // END with no open START (orphan)
                    } else {
                        depth_dbg--;
                    }
                }
            }
            // DIAG: per-hart mapped window. All four harts run CONCURRENTLY for the whole session, so these
            // windows must OVERLAP. If they are disjoint/staggered, the per-hart rdcycle counters are not on a
            // common origin and hart0's calibration cannot be applied to harts 1..3 as-is.
            uint64_t h_min = ~0ull, h_max = 0;
            for (const auto& m : ctx.hz_raw[h]) {
                const uint64_t t = map_ts(m.rdc);
                h_min = std::min(h_min, t);
                h_max = std::max(h_max, t);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart {} ({}): {} zones -> Tracy; window [{:.1f} .. {:.1f}] ms rel, raw_rdc0={}",
                h,
                is_reader ? "READ" : "RELAY",
                nz,
                ctx.hz_raw[h].empty() ? 0.0 : static_cast<double>(h_min - hz_min) / 1.35e6,
                ctx.hz_raw[h].empty() ? 0.0 : static_cast<double>(h_max - hz_min) / 1.35e6,
                ctx.hz_raw[h].empty() ? 0ull : ctx.hz_raw[h].front().rdc);
            // DIAG: where in time this hart's zones actually LIVE (10 equal buckets over the whole hart-zone
            // span). If lanes look "staggered" in the GUI but their min/max windows overlap, the answer is
            // here: a lane whose zones all sit in one bucket renders as one narrow merged box at that offset.
            {
                const uint64_t span = (hz_max > hz_min) ? (hz_max - hz_min) : 1;
                uint32_t bucket[10] = {0};
                for (const auto& m : ctx.hz_raw[h]) {
                    const uint64_t t = map_ts(m.rdc);
                    const uint64_t rel = (t > hz_min) ? (t - hz_min) : 0;
                    uint32_t bi = static_cast<uint32_t>((rel * 10) / span);
                    bucket[std::min<uint32_t>(bi, 9)]++;
                }
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler]   hart {} TIME-HIST(10 buckets): {} {} {} {} {} {} {} {} {} {}",
                    h,
                    bucket[0], bucket[1], bucket[2], bucket[3], bucket[4],
                    bucket[5], bucket[6], bucket[7], bucket[8], bucket[9]);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler]   hart {} BALANCE: starts={} ends={} left_open={} orphan_ends={} max_depth={}",
                h,
                nz,
                n_end,
                depth_dbg,
                unbalanced_dbg,
                max_depth_dbg);
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] hart zones: {} spans over {:.1f} ms (a={:.5f}, {} calib samples kept)",
            total,
            static_cast<double>(hz_max - hz_min) / 1.35e6,
            a,
            nfit);
    }
}

}  // namespace tt::tt_metal
