// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// X280 SPSC prototype host launcher.
//
// Modes:
//   run        - STEP 1: launch the fixed-slot producer (kernels/producer.cpp)
//                and sleep forever.
//   verify     - STEP 1 self-check: read the fixed 8B slot, confirm seq advances
//                at the pace cadence.
//   ring       - STEP 2: launch the flit-ring producer (kernels/ring_producer.cpp)
//                and sleep forever so the X280 3-hart consumer can drain it.
//   ringverify - STEP 2 self-check: launch the ring producer, then act as a slow
//                single consumer from the host -- drain cells in order, check
//                word[0]==word[15]==index (lossless + ordered), and confirm the
//                producer blocked (backpressure) rather than overwriting. No X280
//                needed; this validates the producer + ring protocol.
//
// Build: needs -DBUILD_PROGRAMMING_EXAMPLES=ON.
// Run (single-chip box): env -u TT_MESH_GRAPH_DESC_PATH TT_METAL_SKIP_DRAM_TLBS=1 \
//        ./metal_example_x280_spsc [logical_dev=0] [mode] [arg]
//   verify:     arg = pace_cycles (default 540)
//   ring:       arg = pace_cycles (default 540; 0 = unpaced)
//   ringverify: arg = cells to drain (default 4000); pace forced to 0 (unpaced)
// On a multi-chip box pass the logical id that maps to the X280's PCIe chip.

#include <fmt/ostream.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#if defined(TRACY_ENABLE)
#include <tracy/Tracy.hpp>
#include <tracy/TracyTTDevice.hpp>
#include <common/TracyTTDeviceData.hpp>
#endif

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---- shared L1 layout for the ring (must match kernels/ring_producer.cpp) ----
static constexpr uint32_t kRingBase = 0x80000;
static constexpr uint32_t kRingCells = 32;                       // power of 2
static constexpr uint32_t kWAddr = kRingBase + kRingCells * 64;  // 0x80800
static constexpr uint32_t kRAddr = kWAddr + 64;                  // 0x80840
static constexpr uint32_t kBlockAddr = kRAddr + 64;              // 0x80880

static uint32_t read_u32(IDevice* dev, const CoreCoord& core, uint32_t addr) {
    std::vector<uint32_t> v;
    detail::ReadFromDeviceL1(dev, core, addr, 4, v, tt::CoreType::WORKER);
    return v[0];
}

static void write_u32(IDevice* dev, const CoreCoord& core, uint32_t addr, uint32_t val) {
    std::vector<uint32_t> v{val};
    detail::WriteToDeviceL1(dev, core, addr, v, tt::CoreType::WORKER);
}

// ---- decoded zone marker (device ticks + name-hash + START/END) ----
struct ZoneMarker {
    uint64_t ts;       // device wall-clock ticks (AICLK)
    uint32_t zone_id;  // 16-bit name hash
    uint32_t type;     // 0 = START, 1 = END
};

// Same FNV-1a/16 the kernel uses, so we can map zone ids back to demo names.
static uint32_t cp_hash16(const std::string& s) {
    uint32_t b = 2166136261u;
    for (unsigned char c : s) {
        b = (b ^ c) * 16777619u;
    }
    return ((b & 0xFFFF) ^ ((b >> 16) & 0xFFFF)) & 0xFFFF;
}

// ---- Native Tracy emission (same TracyTT* API as RealtimeProfilerTracyHandler) ----
// Streams the decoded device zones to a connected Tracy capture (tracy-capture or
// the GUI): create a TT device context, populate it with the device frequency,
// then push ZONE_START/ZONE_END TTDeviceMarkers. Markers only flow while a Tracy
// client is connected, so we wait for the capture to attach first.
static void emit_tracy(const std::vector<ZoneMarker>& mks, const CoreCoord& noc0, double freq_ghz) {
#if defined(TRACY_ENABLE)
    if (mks.empty()) {
        return;
    }
    const std::map<uint32_t, std::string> names = {{cp_hash16("outer"), "outer"}, {cp_hash16("inner"), "inner"}};

    fmt::print("waiting up to 30s for a Tracy capture to connect (run tracy-capture -o out.tracy)...\n");
    std::fflush(stdout);
    for (int i = 0; i < 300 && !tracy::GetProfiler().IsConnected(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!tracy::GetProfiler().IsConnected()) {
        fmt::print("no Tracy capture connected; skipping emit (re-run with tracy-capture attached)\n");
        return;
    }

    TracyTTCtx ctx = TracyTTContext();
    // tgpu = first device tick (anchor); tcpu = 0 -> Tracy uses ctx-creation host time.
    TracyTTContextPopulate(ctx, 0, static_cast<double>(mks.front().ts), freq_ghz);
    const std::string ctxname = fmt::format("X280 pull (Tensix {},{})", noc0.x, noc0.y);
    TracyTTContextName(ctx, ctxname.c_str(), ctxname.size());

    uint64_t pushed = 0;
    for (const auto& m : mks) {
        tracy::TTDeviceMarker marker;
        marker.chip_id = 0;
        marker.core_x = noc0.x;
        marker.core_y = noc0.y;
        marker.risc = tracy::RiscType::BRISC;
        marker.timestamp = m.ts;
        marker.runtime_host_id = 0;
        auto it = names.find(m.zone_id);
        marker.marker_name = (it != names.end()) ? it->second : fmt::format("zone_0x{:x}", m.zone_id);
        marker.file = "continous_profiler";
        marker.line = 0;
        if (m.type == 0) {
            marker.marker_type = tracy::TTDeviceMarkerType::ZONE_START;
            TracyTTPushStartMarker(ctx, marker);
        } else {
            marker.marker_type = tracy::TTDeviceMarkerType::ZONE_END;
            TracyTTPushEndMarker(ctx, marker);
        }
        pushed++;
    }
    fmt::print("emitted {} Tracy device markers; flushing to capture...\n", pushed);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(2));  // let Tracy flush the serial queue
    TracyTTDestroy(ctx);
#else
    (void)mks;
    (void)noc0;
    (void)freq_ghz;
    fmt::print("built without TRACY_ENABLE; cannot emit to Tracy\n");
#endif
}

// ---- STEP 1 fixed-slot verify ----
static void verify_slot(IDevice* dev, const CoreCoord& core, uint32_t buf_addr, uint32_t pace_cycles) {
    constexpr int kSamples = 20000;
    uint32_t last_seq = 0, last_ts = 0;
    bool have_last = false;
    int records = 0;
    uint64_t sum_delta = 0;
    uint32_t min_delta = 0xffffffff, max_delta = 0, first_seq = 0, final_seq = 0;
    for (int i = 0; i < kSamples && records < 64; i++) {
        std::vector<uint32_t> rec;
        detail::ReadFromDeviceL1(dev, core, buf_addr, 8, rec, tt::CoreType::WORKER);
        const uint32_t seq = rec[0], ts = rec[1];
        if (seq == 0) {
            continue;
        }
        if (!have_last) {
            have_last = true;
            first_seq = seq;
        } else if (seq != last_seq) {
            const uint32_t dseq = seq - last_seq;
            const uint32_t per = (ts - last_ts) / (dseq ? dseq : 1);
            sum_delta += per;
            min_delta = std::min(min_delta, per);
            max_delta = std::max(max_delta, per);
            records++;
        }
        last_seq = seq;
        last_ts = ts;
        final_seq = seq;
    }
    fmt::print("\n==================== VERIFY (fixed slot) ====================\n");
    if (records == 0) {
        fmt::print("FAIL: never observed a changing seq\n");
        return;
    }
    const uint32_t avg = static_cast<uint32_t>(sum_delta / records);
    fmt::print("seq advanced {} -> {} ({} transitions)\n", first_seq, final_seq, records);
    fmt::print(
        "wall-clock ticks/record: avg {} min {} max {} (pace {} = ~{} ns)\n",
        avg,
        min_delta,
        max_delta,
        pace_cycles,
        (avg * 1000 + 675) / 1350);
    const bool ok = final_seq > first_seq && avg >= pace_cycles / 2 && avg <= pace_cycles * 3;
    fmt::print("{}\n", ok ? "PASS: producer advances at ~pace cadence" : "WARN: cadence outside band");
}

// ---- STEP 2 host-side ring consumer (single, slow) ----
static void verify_ring(IDevice* dev, const CoreCoord& core, uint32_t cells_to_drain) {
    fmt::print("\n==================== VERIFY (flit ring) ====================\n");
    fmt::print("host acting as single slow consumer; draining {} cells in order...\n", cells_to_drain);
    uint32_t r = 0;          // next index to consume
    uint64_t errors = 0;     // integrity / ordering violations
    uint32_t first_err = 0;  // first bad index
    bool had_err = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (r < cells_to_drain) {
        const uint32_t w = read_u32(dev, core, kWAddr);
        if (w == r) {
            continue;  // nothing new yet
        }
        for (; r < w && r < cells_to_drain; r++) {
            const uint32_t slot = r & (kRingCells - 1);
            std::vector<uint32_t> cell;
            detail::ReadFromDeviceL1(dev, core, kRingBase + slot * 64, 64, cell, tt::CoreType::WORKER);
            // word[0] and word[15] must both equal the absolute index r.
            if (cell[0] != r || cell[15] != r) {
                if (!had_err) {
                    had_err = true;
                    first_err = r;
                    fmt::print("  ERROR at index {}: word[0]={} word[15]={} (expected {})\n", r, cell[0], cell[15], r);
                }
                errors++;
            }
        }
        write_u32(dev, core, kRAddr, r);  // publish read pointer -> frees the producer
    }
    const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint32_t blocked = read_u32(dev, core, kBlockAddr);
    const uint32_t final_w = read_u32(dev, core, kWAddr);
    fmt::print("drained {} cells in {:.2f} s ({:.0f} cells/s)\n", r, dt, r / dt);
    fmt::print("producer backpressure events (ring went full): {}\n", blocked);
    fmt::print("final producer w = {}\n", final_w);
    fmt::print("integrity/order errors: {}{}\n", errors, had_err ? fmt::format(" (first at index {})", first_err) : "");
    const bool ok = errors == 0 && r == cells_to_drain && blocked > 0;
    fmt::print(
        "{}\n",
        ok ? "PASS: lossless + in-order, and backpressure exercised (slow consumer never lost a cell)"
           : (errors ? "FAIL: data loss/reorder detected" : "WARN: completed but backpressure not observed"));
}

// ---- STEP 4 host-side marker decoder (continuous profiler) ----
// Drains the SPSC ring and decodes 8B markers (8 per flit), validating the
// kernel_profiler-format fields the X280 consumer would otherwise process.
static void verify_zones(IDevice* dev, const CoreCoord& core, const CoreCoord& noc0, uint32_t markers_to_drain) {
    fmt::print("\n==================== VERIFY (continuous profiler zones) ====================\n");
    fmt::print("host draining + decoding {} markers (8B each, 8 per flit) in order...\n", markers_to_drain);
    uint32_t r = 0;  // next flit index to consume
    uint64_t markers = 0, starts = 0, ends = 0, invalid = 0, ts_backwards = 0;
    uint64_t last_ts = 0;
    bool have_last = false;
    std::vector<ZoneMarker> trace;
    trace.reserve(markers_to_drain);
    const auto t0 = std::chrono::steady_clock::now();
    while (markers < markers_to_drain) {
        const uint32_t w = read_u32(dev, core, kWAddr);
        if (w == r) {
            continue;
        }
        for (; r < w && markers < markers_to_drain; r++) {
            const uint32_t slot = r & (kRingCells - 1);
            std::vector<uint32_t> flit;
            detail::ReadFromDeviceL1(dev, core, kRingBase + slot * 64, 64, flit, tt::CoreType::WORKER);
            for (uint32_t mslot = 0; mslot < 8; mslot++) {
                const uint32_t w0 = flit[mslot * 2], w1 = flit[mslot * 2 + 1];
                if ((w0 & 0x80000000u) == 0) {
                    invalid++;
                }
                const uint32_t timer_id = (w0 >> 12) & 0x7FFFF;
                const uint32_t zone_id = timer_id & 0xFFFF;
                const uint32_t type = (timer_id >> 16) & 0x7;
                const uint64_t ts = (static_cast<uint64_t>(w0 & 0xFFF) << 32) | w1;
                if (type == 0) {
                    starts++;
                } else if (type == 1) {
                    ends++;
                }
                if (have_last && ts < last_ts) {
                    ts_backwards++;
                }
                last_ts = ts;
                have_last = true;
                markers++;
                trace.push_back({ts, zone_id, type});
            }
        }
        write_u32(dev, core, kRAddr, r);
    }
    const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint32_t blocked = read_u32(dev, core, kBlockAddr);
    fmt::print("decoded {} markers in {:.2f} s ({:.0f} markers/s)\n", markers, dt, markers / dt);
    fmt::print("  ZONE_START {} | ZONE_END {} (diff {})\n", starts, ends, static_cast<int64_t>(starts - ends));
    fmt::print("  invalid (no valid bit): {} | timestamp-went-backwards: {}\n", invalid, ts_backwards);
    fmt::print("  producer backpressure events: {}\n", blocked);
    const bool ok = invalid == 0 && ts_backwards == 0 && (starts >= ends) && (starts - ends) <= 2;
    fmt::print(
        "{}\n",
        ok ? "PASS: well-formed markers, monotonic timestamps, START/END balanced (lossless stream)"
           : "FAIL: malformed markers / non-monotonic timestamps");
    emit_tracy(trace, noc0, 1.349987);
}

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    const std::string mode = argc > 2 ? argv[2] : "run";

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    IDevice* dev = mesh_device->get_devices().at(0);
    Program program = CreateProgram();
    const CoreCoord producer_logical{0, 0};
    const CoreCoord noc0 = dev->worker_core_from_logical_core(producer_logical);

    // ---- STEP 5: FULL PIPELINE -- Tensix L1 ring -> X280 pull -> D2H push ->
    // host pinned FIFO. Host creates a D2H socket (host pinned FIFO + config in
    // (0,0) L1), launches the zone producer, then read()s decoded markers from
    // the FIFO. The X280 `x280_pipe` relays flits from the ring into the FIFO.
    if (mode == "zonepipe") {
        const uint32_t n_iters = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 1000u;
        const uint32_t work_cycles = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 100u;
        const uint32_t n_flits = n_iters / 2;      // 8 markers/flit, 4 markers/iter
        constexpr uint32_t fifo_size = 64 * 1024;  // 1024 x 64B pages in host pinned mem
        constexpr uint32_t page_size = 64;         // one 64B flit per page

        // D2H socket bound to (0,0); its config buffer lives in (0,0) L1 so the
        // X280 can read the host FIFO target over the NoC.
        distributed::MeshCoreCoord sender_core{distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)};
        distributed::D2HSocket socket(mesh_device, sender_core, fifo_size);
        socket.set_page_size(page_size);
        const uint32_t cfg_addr = socket.get_config_buffer_address();

        CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "x280_spsc/kernels/zone_demo.cpp",
            producer_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {
                    {"CP_RING_BASE", fmt::format("{:#x}", kRingBase)},
                    {"CP_RING_CELLS", std::to_string(kRingCells)},
                    {"CP_W_ADDR", fmt::format("{:#x}", kWAddr)},
                    {"CP_R_ADDR", fmt::format("{:#x}", kRAddr)},
                    {"CP_BLOCK_ADDR", fmt::format("{:#x}", kBlockAddr)},
                    {"N_ITERS", std::to_string(n_iters)},
                    {"WORK_CYCLES", std::to_string(work_cycles)},
                }});

        fmt::print("\n==================== PIPELINE TARGET (for X280 x280_pipe) ====================\n");
        fmt::print(
            "Tensix NOC0 ({},{}) | ring {:#x} W={:#x} R={:#x} | socket config_addr={:#x}\n",
            noc0.x,
            noc0.y,
            kRingBase,
            kWAddr,
            kRAddr,
            cfg_addr);
        fmt::print(
            "host FIFO {} B, page {} B; producing {} iters -> {} flits -> {} markers\n",
            fifo_size,
            page_size,
            n_iters,
            n_flits,
            n_flits * 8);
        // The hardcoded ring (kRingBase..kBlockAddr+4) must not overlap the
        // allocator-placed socket config buffer.
        if (cfg_addr + distributed::D2HSocket::required_config_buffer_size() > kRingBase &&
            cfg_addr < kBlockAddr + 64) {
            fmt::print(
                "WARNING: socket config_addr {:#x} overlaps ring region {:#x}..{:#x}!\n",
                cfg_addr,
                kRingBase,
                kBlockAddr + 64);
        }
        fmt::print("run:  ./x280_pipe {} {} {:#x}\n", noc0.x, noc0.y, cfg_addr);
        fmt::print("=============================================================================\n");

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        fmt::print("zone producer launched; reading {} markers from host pinned FIFO...\n", n_flits * 8);
        std::fflush(stdout);

        // Drain the host FIFO page-by-page and decode markers (8 per 64B page).
        std::vector<uint32_t> page(page_size / sizeof(uint32_t));
        std::vector<ZoneMarker> trace;
        trace.reserve(n_flits * 8);
        uint64_t markers = 0, starts = 0, ends = 0, invalid = 0, ts_backwards = 0, last_ts = 0;
        bool have_last = false;
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t p = 0; p < n_flits; p++) {
            socket.read(page.data(), 1, /*notify_sender=*/true);
            for (uint32_t m = 0; m < 8; m++) {
                const uint32_t w0 = page[m * 2], w1 = page[m * 2 + 1];
                if ((w0 & 0x80000000u) == 0) {
                    invalid++;
                }
                const uint32_t timer_id = (w0 >> 12) & 0x7FFFF;
                const uint32_t zone_id = timer_id & 0xFFFF;
                const uint32_t type = (timer_id >> 16) & 0x7;
                const uint64_t ts = (static_cast<uint64_t>(w0 & 0xFFF) << 32) | w1;
                if (type == 0) {
                    starts++;
                } else if (type == 1) {
                    ends++;
                }
                if (have_last && ts < last_ts) {
                    ts_backwards++;
                }
                last_ts = ts;
                have_last = true;
                markers++;
                trace.push_back({ts, zone_id, type});
            }
        }
        const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fmt::print("\n==================== PIPELINE RESULT (host pinned memory) ====================\n");
        fmt::print(
            "received {} markers ({} flits) in {:.2f} s ({:.0f} markers/s) via L1->X280->D2H->host\n",
            markers,
            n_flits,
            dt,
            markers / dt);
        fmt::print("  ZONE_START {} | ZONE_END {} (diff {})\n", starts, ends, static_cast<int64_t>(starts - ends));
        fmt::print("  invalid: {} | timestamp-went-backwards: {}\n", invalid, ts_backwards);
        const bool ok = invalid == 0 && ts_backwards == 0 && starts == ends && markers == n_flits * 8;
        fmt::print("{}\n", ok ? "PASS: full pipeline delivered all markers losslessly to host memory" : "FAIL");
        emit_tracy(trace, noc0, 1.349987);
        return 0;
    }

    // ---- STEP 4: continuous-profiler ZONES -- DeviceZoneScopedN-style scopes
    // whose ctor/dtor stream START/END markers into the SPSC ring (8 markers per
    // 64B flit). `zoneverify` decodes the markers from the host; `zones` runs
    // forever for the X280 consumer to drain.
    if (mode == "zones" || mode == "zoneverify") {
        const uint32_t n_iters = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 1000u;
        const uint32_t work_cycles = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 100u;
        // 4 markers/iter, 8 markers/flit -> (n_iters/2) full flits are published.
        const uint32_t markers_to_drain = (n_iters / 2) * 8;
        CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "x280_spsc/kernels/zone_demo.cpp",
            producer_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {
                    {"CP_RING_BASE", fmt::format("{:#x}", kRingBase)},
                    {"CP_RING_CELLS", std::to_string(kRingCells)},
                    {"CP_W_ADDR", fmt::format("{:#x}", kWAddr)},
                    {"CP_R_ADDR", fmt::format("{:#x}", kRAddr)},
                    {"CP_BLOCK_ADDR", fmt::format("{:#x}", kBlockAddr)},
                    {"N_ITERS", std::to_string(n_iters)},
                    {"WORK_CYCLES", std::to_string(work_cycles)},
                }});
        fmt::print("\n==================== ZONES TARGET (for X280) ====================\n");
        fmt::print(
            "device {}: continuous-profiler kernel on BRISC of (0,0) = NOC0 ({},{}); {} iters -> {} flits\n",
            device_id,
            noc0.x,
            noc0.y,
            n_iters,
            n_iters / 2);
        fmt::print(
            "ring {:#x} ({} flits x 64B), W={:#x} R={:#x} BLK={:#x}; 8 markers (8B) per flit\n",
            kRingBase,
            kRingCells,
            kWAddr,
            kRAddr,
            kBlockAddr);
        fmt::print("marker: word0=0x80000000|((zone_id|type<<16)<<12)|wall_hi12, word1=wall_lo (type 0=START 1=END)\n");
        fmt::print("=================================================================\n");
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        fmt::print("zone kernel launched (mode={})\n", mode);
        std::fflush(stdout);
        if (mode == "zoneverify") {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            verify_zones(dev, producer_logical, noc0, markers_to_drain);
            return 0;
        }
        fmt::print("host sleeping (Ctrl-C to stop); attach the X280 consumer now\n");
        std::fflush(stdout);
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(60));
        }
        return 0;
    }

    // ---- STEP 3: GRID mode -- ring producer on EVERY worker core ----
    // Each core runs an independent flit-ring in its own L1 (same fixed layout,
    // since each core has its own L1). The X280 drains the grid with one hart per
    // column-band of cores (spatially separated -> independent NIUs/routes).
    if (mode == "grid") {
        const uint32_t pace_cycles = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 540u;
        const CoreCoord grid = dev->compute_with_storage_grid_size();
        const CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});
        CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "x280_spsc/kernels/ring_producer.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {
                    {"RING_BASE", fmt::format("{:#x}", kRingBase)},
                    {"RING_CELLS", std::to_string(kRingCells)},
                    {"W_ADDR", fmt::format("{:#x}", kWAddr)},
                    {"R_ADDR", fmt::format("{:#x}", kRAddr)},
                    {"BLOCK_ADDR", fmt::format("{:#x}", kBlockAddr)},
                    {"PACE_CYCLES", std::to_string(pace_cycles)},
                }});
        fmt::print("\n==================== GRID RING TARGET (for X280) ====================\n");
        fmt::print(
            "device {}: ring producer on ALL {}x{} worker cores; pace={} cycles{}\n",
            device_id,
            grid.x,
            grid.y,
            pace_cycles,
            pace_cycles == 0 ? " (unpaced)" : "");
        fmt::print(
            "per-core L1: RING_BASE={:#x} ({} x 64B), W={:#x} R={:#x} BLK={:#x}\n",
            kRingBase,
            kRingCells,
            kWAddr,
            kRAddr,
            kBlockAddr);
        fmt::print("cell[idx]: word[0]=word[15]=idx (per-core seq), word[1]=wall-clock-lo\n");
        // Emit every worker core's NOC0 coordinate for the consumer's coordfile.
        for (uint32_t ly = 0; ly < grid.y; ly++) {
            for (uint32_t lx = 0; lx < grid.x; lx++) {
                const CoreCoord c = dev->worker_core_from_logical_core(CoreCoord{lx, ly});
                fmt::print("CORE {} {}\n", c.x, c.y);
            }
        }
        fmt::print("=====================================================================\n");
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        fmt::print("grid ring producer launched (mode=grid)\n");
        fmt::print("host sleeping (Ctrl-C to stop); attach the X280 grid consumer now\n");
        std::fflush(stdout);
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(60));
        }
        return 0;
    }

    const bool ring_mode = (mode == "ring" || mode == "ringverify");

    if (!ring_mode) {
        // ---- STEP 1: fixed 8B slot producer ----
        const uint32_t pace_cycles = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 540u;
        constexpr uint32_t buf_addr = 0x80000;
        CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "x280_spsc/kernels/producer.cpp",
            producer_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = {
                    {"BUF_ADDR", fmt::format("{:#x}", buf_addr)}, {"PACE_CYCLES", std::to_string(pace_cycles)}}});
        fmt::print("device {}: fixed-slot producer on BRISC of (0,0) = NOC0 ({},{})\n", device_id, noc0.x, noc0.y);
        fmt::print("record: 8B @ L1 {:#x}; pace={} cycles\n", buf_addr, pace_cycles);
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        fmt::print("producer launched (mode={})\n", mode);
        std::fflush(stdout);
        if (mode == "verify") {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            verify_slot(dev, producer_logical, buf_addr, pace_cycles);
            return 0;
        }
        fmt::print("host sleeping (Ctrl-C to stop)\n");
        std::fflush(stdout);
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(60));
        }
        return 0;
    }

    // ---- STEP 2: flit-ring producer ----
    const uint32_t pace_cycles =
        (mode == "ringverify") ? 0u : (argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 540u);
    const uint32_t cells_to_drain =
        (mode == "ringverify") ? (argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 4000u) : 0u;

    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "x280_spsc/kernels/ring_producer.cpp",
        producer_logical,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .defines = {
                {"RING_BASE", fmt::format("{:#x}", kRingBase)},
                {"RING_CELLS", std::to_string(kRingCells)},
                {"W_ADDR", fmt::format("{:#x}", kWAddr)},
                {"R_ADDR", fmt::format("{:#x}", kRAddr)},
                {"BLOCK_ADDR", fmt::format("{:#x}", kBlockAddr)},
                {"PACE_CYCLES", std::to_string(pace_cycles)},
            }});

    fmt::print("\n==================== RING TARGET (for X280) ====================\n");
    fmt::print("device {}: ring producer on BRISC of (0,0) = Tensix NOC0 ({},{})\n", device_id, noc0.x, noc0.y);
    fmt::print("RING_BASE  = {:#x}  ({} cells x 64B = {} B)\n", kRingBase, kRingCells, kRingCells * 64);
    fmt::print("W_ADDR     = {:#x}  (producer writes, X280 reads)\n", kWAddr);
    fmt::print("R_ADDR     = {:#x}  (X280 writes, producer reads)\n", kRAddr);
    fmt::print("BLOCK_ADDR = {:#x}  (producer backpressure-event count)\n", kBlockAddr);
    fmt::print("pace       = {} cycles{}\n", pace_cycles, pace_cycles == 0 ? " (unpaced / max rate)" : "");
    fmt::print("cell[idx]: word[0]=word[15]=idx (seq), word[1]=wall-clock-lo\n");
    fmt::print("================================================================\n");

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    fmt::print("ring producer launched (mode={})\n", mode);
    std::fflush(stdout);

    if (mode == "ringverify") {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        verify_ring(dev, producer_logical, cells_to_drain);
        return 0;
    }

    fmt::print("host sleeping (Ctrl-C to stop); attach the X280 consumer now\n");
    std::fflush(stdout);
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
    return 0;
}
