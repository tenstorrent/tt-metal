// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: dispatches kernels that emit 10 differently-named DeviceZoneScopedN zones
// (with increasing durations) on ALL 5 RISCs of a small core grid, then closes. It does NOT drive the drainer
// itself -- run it with TT_METAL_PERF_DEBUG_PROFILER=1 so the PerfDebugProfiler boots at MeshDevice bring-up
// and captures these zones (verify with a connected tracy-capture). TT_METAL_DEVICE_PROFILER=1 enables the
// device profiler so the kernels actually emit markers. Grid + iteration count are overridable via argv.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // --delay is the KNEE knob: uniform nop-iterations per zone, the SAME unit as the standalone drain harness
    // --proddelay, so 0 means MAX RATE (no spin) exactly as it does there. Smaller = higher marker rate.
    // Omitting --delay entirely selects the graduated ~1..100 us wall-clock durations, which is the right
    // default for a representative capture -- graduated is a separate MODE, not a magic --delay value.
    uint32_t gx = 2, gy = 2, n_iters = 50, zone_cyc = 0;  // small grid + modest iters keep the run quick
    bool knee_mode = false;                               // set by --delay, including --delay 0
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
            knee_mode = true;  // NOT `zone_cyc != 0`: --delay 0 is a real knee point (max rate)
        }
    }

    // TT_METAL_SLOW_DISPATCH_MODE=1 takes the whole run off the command queue. Needed by the Tensix-BRISC
    // drainer control (TT_METAL_PERF_DEBUG_DRAIN_TENSIX), which parks a resident program on a worker core --
    // something fast dispatch will not allow. The workload itself is unchanged; only how it is launched is.
    const char* sd = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const bool slow_dispatch = sd != nullptr && *sd != '\0' && *sd != '0';

    // TT_METAL_PERF_DEBUG_NUM_CQS selects the hardware command-queue count. It exists to test whether the
    // DRISC PCIe hang scales with how much dispatch traffic shares the PCIe tile with the drainer's egress:
    // each CQ adds its own completion queue and its own dispatch cores driving that tile. Default 1 matches
    // every measurement taken before this knob existed, so leaving it unset changes nothing.
    const char* nq = std::getenv("TT_METAL_PERF_DEBUG_NUM_CQS");
    const size_t num_cqs = (nq != nullptr && *nq != '\0') ? (size_t)std::strtoul(nq, nullptr, 10) : 1;

    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(
        device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, num_cqs);
    Program program = CreateProgram();

    // Clamp the requested grid to the device's compute grid; --gx 0 / --gy 0 (or an over-large value)
    // means "use the full grid". Passing a CoreRange past the grid would throw.
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
        {"ZONE_MODE", knee_mode ? "1" : "0"},
        {"ZONE_CYC", std::to_string(zone_cyc) + "u"}};
    const std::string kdir = "tt_metal/programming_examples/profiler/test_perf_debug_zones/kernels/";

    // BRISC (RISCV_0) + NCRISC (RISCV_1): the data-movement zone kernel (tags BR_/NC_).
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
    // TRISC0/1/2: the compute zone kernel (tags T0_/T1_/T2_).
    CreateKernel(program, kdir + "zones_compute.cpp", cores, ComputeConfig{.defines = defs});

    // Report the offered load so a knee sweep is self-documenting: 2 markers per zone (START+END), 10 zones
    // per iteration, 5 RISCs per core. Rate uses the ~1.35 GHz boosted aiclk the spin loop counts against.
    const uint32_t lanes = gx * gy * 5;
    const uint64_t markers = (uint64_t)lanes * 10ull * 2ull * (uint64_t)n_iters;
    printf(
        "[perf-debug zones] dispatching %ux%u cores x 5 RISCs x 10 named zones x %u iters\n"
        "[perf-debug zones]   lanes=%u  total markers=%llu  --delay=%u (%s)\n",
        gx,
        gy,
        n_iters,
        lanes,
        (unsigned long long)markers,
        zone_cyc,
        knee_mode ? "uniform nop-spin: knee mode" : "graduated ~1..100us wall-clock");
    if (knee_mode) {
        // No tick-derived rate here on purpose: --delay is nop LOOP ITERATIONS over a volatile counter
        // (same unit as the standalone drain harness --proddelay), and one iteration is several cycles, so markers/s
        // cannot be derived from it without measuring cycles-per-iteration. Printing a tick-derived rate is
        // what previously made this knee look ~30x worse than the harness's.
        printf(
            "[perf-debug zones]   --delay=%u nop-iterations/zone (same unit as the standalone drain harness "
            "--proddelay; 0 = max rate)\n",
            zone_cyc);
    }
    if (slow_dispatch) {
        IDevice* device = mesh_device->get_devices().front();
        detail::CompileProgram(device, program);
        detail::WriteRuntimeArgsToDevice(device, program);
        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    } else {
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
    }
    printf("[perf-debug zones] workload done; closing device.\n");
    mesh_device->close();
    return 0;
}
