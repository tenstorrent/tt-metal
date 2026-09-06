// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming-profiler workload: kernels emit 10 differently-named DeviceZoneScopedN zones of increasing duration
// on all 5 RISCs of a small grid. Run with TT_METAL_STREAMING_PROFILER=1 (its own mode; do not combine it with
// the mutually exclusive TT_METAL_DEVICE_PROFILER); add TT_METAL_STREAMING_PROFILER_TRACY=1 to check against a
// connected tracy-capture. Grid and iterations via argv.
//
// --bench K prices one marker kind on the device instead: K = 0 spin only, 1 empty DeviceZoneScopedN, 2 DeviceFlag,
// 3 DeviceTimestampedData. Each RISC times bursts against its wall clock and the host prints cycles per marker; the
// K = 0 run with the same --benchdelay is the loop's own cost to subtract. Measure on a 1x1 grid with a paced
// --benchdelay (20 is enough) and no "profiler stalls" or "FAILED TO START" in the log, or the number is the stall.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // --delay sets uniform nop-iterations per zone; 0 is a valid setting meaning max rate. Omitting it
    // selects a separate mode: graduated ~1..100 us wall-clock zone durations.
    uint32_t gx = 2, gy = 2, n_iters = 50, zone_cyc = 0;
    bool knee_mode = false;     // set by --delay, including --delay 0
    uint32_t emit_markers = 0;  // --markers 1: emit the point-marker trio (Flag/Data/Iter) per iteration
    bool bench = false;         // --bench K: the marker-cost microbench, K the marker kind
    uint32_t bench_kind = 0, bench_delay = 0;
    constexpr uint32_t kBenchAddr = 0x170000;  // L1 scratch the bench kernels leave {cycles, markers} per RISC in
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        uint32_t v = (uint32_t)std::strtoul(argv[i + 1], nullptr, 10);
        if (a == "--gx") {
            gx = v;
        } else if (a == "--gy") {
            gy = v;
        } else if (a == "--iters") {
            n_iters = v;
        } else if (a == "--delay") {
            zone_cyc = v;
            knee_mode = true;  // not `zone_cyc != 0`: --delay 0 is a real knee point (max rate)
        } else if (a == "--markers") {
            emit_markers = v;
        } else if (a == "--bench") {
            bench = true;
            bench_kind = v;
        } else if (a == "--benchdelay") {
            bench_delay = v;
        }
    }

    // A counting subscriber: the capture is decoded and totalled even with no sink armed.
    struct Totals {
        std::atomic<uint64_t> zones{0}, points{0}, stalls{0};
    } totals;
    using experimental::streaming_profiler::Batch;
    using experimental::streaming_profiler::Channel;
    const auto sub = experimental::streaming_profiler::Subscribe("zones-example", [&](const Batch<Channel::All>& b) {
        totals.zones += b.zones.size();
        totals.points += b.events.size() + b.timestamped_data.size();
        totals.stalls += b.stall_count;
    });

    const char* sd = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const bool slow_dispatch = sd != nullptr && *sd != '\0' && *sd != '0';

    int device_id = 0;
    // TT_METAL_STREAMING_PROFILER_FULL_MESH=RxC opens the whole mesh in one process: N devices, one profiler boot.
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    if (const char* fm = std::getenv("TT_METAL_STREAMING_PROFILER_FULL_MESH"); fm != nullptr && *fm != '\0') {
        uint32_t rows = (uint32_t)std::strtoul(fm, nullptr, 10);
        const char* xp = std::strchr(fm, 'x');
        uint32_t cols = xp != nullptr ? (uint32_t)std::strtoul(xp + 1, nullptr, 10) : 1;
        mesh_device = distributed::MeshDevice::create(
            distributed::MeshDeviceConfig(distributed::MeshShape(rows, cols)),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            /*num_command_queues=*/1);
    } else {
        mesh_device = distributed::MeshDevice::create_unit_mesh(
            device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, /*num_command_queues=*/1);
    }
    Program program = CreateProgram();

    // --gx 0 / --gy 0, or an over-large value, means the full grid; a CoreRange past the grid would throw.
    CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (gx == 0 || gx > grid.x) {
        gx = grid.x;
    }
    if (gy == 0 || gy > grid.y) {
        gy = grid.y;
    }
    CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
    std::map<std::string, std::string> defs{
        {"N_ITERS", std::to_string(n_iters) + "u"},
        {"ZONE_MODE", bench ? "2" : (knee_mode ? "1" : "0")},
        {"EMIT_MARKERS", emit_markers != 0 ? "1" : "0"},
        {"ZONE_CYC", std::to_string(zone_cyc) + "u"},
        {"BENCH_KIND", std::to_string(bench_kind)},
        {"BENCH_DELAY", std::to_string(bench_delay)},
        {"BENCH_ADDR", std::to_string(kBenchAddr) + "u"}};
    const std::string kdir = "tt_metal/programming_examples/profiler/test_streaming_profiler_zones/kernels/";

    CreateKernel(
        program,
        kdir + "zones_dm.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defs});
    CreateKernel(
        program,
        kdir + "zones_dm.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defs});
    CreateKernel(program, kdir + "zones_compute.cpp", cores, ComputeConfig{.defines = defs});

    // Offered load: 2 markers per zone (start + end), 10 zones per iteration, 5 RISCs per core.
    const uint32_t lanes = gx * gy * 5;
    const uint64_t markers = (uint64_t)lanes * 10ull * 2ull * (uint64_t)n_iters;
    printf(
        "[streaming profiler zones] dispatching %ux%u cores x 5 RISCs x 10 named zones x %u iters\n"
        "[streaming profiler zones]   lanes=%u  total markers=%llu  --delay=%u (%s)\n",
        gx,
        gy,
        n_iters,
        lanes,
        (unsigned long long)markers,
        zone_cyc,
        knee_mode ? "uniform nop-spin: knee mode" : "graduated ~1..100us wall-clock");
    if (knee_mode) {
        // Measured on Blackhole: the volatile nop loop costs 10 cycles/iteration (slope 7.407 ns/unit at
        // 1.35 GHz) and a zone's fixed cost is 86.7 ns, fitted over delay 15/500/2000.
        const double zone_ns = 86.7 + zone_cyc * 10.0 / 1.35;
        printf(
            "[streaming profiler zones]   --delay=%u nop-iterations/zone ~= %.0f ns/zone, ~%.2f Mmarkers/s/lane "
            "unthrottled (10 cyc/iteration + ~87 ns/zone measured; same unit as --proddelay; 0 = max rate)\n",
            zone_cyc,
            zone_ns,
            2000.0 / zone_ns);
    }
    // Producer-side wall clock, independent of the receiver's decoded-marker zone window.
    const auto t_launch = std::chrono::steady_clock::now();
    if (slow_dispatch) {
        for (IDevice* device : mesh_device->get_devices()) {
            detail::CompileProgram(device, program);
            detail::WriteRuntimeArgsToDevice(device, program);
            detail::LaunchProgram(device, program, /*wait_until_cores_done=*/false);
        }
        for (IDevice* device : mesh_device->get_devices()) {
            detail::WaitProgramDone(device, program);
        }
    } else {
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
    }
    printf(
        "[streaming profiler zones] workload done in %.1f ms; closing device.\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_launch).count());
    if (bench) {
        static const char* const kRisc[5] = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
        std::vector<uint32_t> slots(10, 0);
        detail::ReadFromDeviceL1(
            mesh_device->get_devices().front(),
            CoreCoord{0, 0},
            kBenchAddr,
            static_cast<uint32_t>(slots.size() * sizeof(uint32_t)),
            slots);
        for (uint32_t slot = 0; slot < 5; slot++) {
            const uint32_t cycles = slots[2 * slot], markers = slots[2 * slot + 1];
            if (markers != 0) {
                printf(
                    "[zonebench] kind %u %-6s %u markers, %u cycles, %.2f cycles/marker (spin included)\n",
                    bench_kind,
                    kRisc[slot],
                    markers,
                    cycles,
                    static_cast<double>(cycles) / markers);
            }
        }
    }
    mesh_device->close();
    experimental::streaming_profiler::Unsubscribe(sub);
    printf(
        "[streaming profiler zones] subscriber saw %llu zones, %llu points, %llu stalls\n",
        (unsigned long long)totals.zones.load(),
        (unsigned long long)totals.points.load(),
        (unsigned long long)totals.stalls.load());
    return 0;
}
