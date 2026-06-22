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
#include <string>
#include <thread>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

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
static void verify_zones(IDevice* dev, const CoreCoord& core, uint32_t markers_to_drain) {
    fmt::print("\n==================== VERIFY (continuous profiler zones) ====================\n");
    fmt::print("host draining + decoding {} markers (8B each, 8 per flit) in order...\n", markers_to_drain);
    uint32_t r = 0;  // next flit index to consume
    uint64_t markers = 0, starts = 0, ends = 0, invalid = 0, ts_backwards = 0;
    uint64_t last_ts = 0;
    bool have_last = false;
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

    // ---- STEP 4: continuous-profiler ZONES -- DeviceZoneScopedN-style scopes
    // whose ctor/dtor stream START/END markers into the SPSC ring (8 markers per
    // 64B flit). `zoneverify` decodes the markers from the host; `zones` runs
    // forever for the X280 consumer to drain.
    if (mode == "zones" || mode == "zoneverify") {
        const uint32_t work_cycles = (mode == "zones" && argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 100u;
        const uint32_t markers_to_drain =
            (mode == "zoneverify") ? (argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 32000u) : 0u;
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
                    {"WORK_CYCLES", std::to_string(work_cycles)},
                }});
        fmt::print("\n==================== ZONES TARGET (for X280) ====================\n");
        fmt::print(
            "device {}: continuous-profiler kernel on BRISC of (0,0) = NOC0 ({},{})\n", device_id, noc0.x, noc0.y);
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
            verify_zones(dev, producer_logical, markers_to_drain);
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
