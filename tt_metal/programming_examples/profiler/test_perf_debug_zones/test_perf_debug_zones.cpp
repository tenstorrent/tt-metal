// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// perf-debug profiler workload: dispatches kernels that emit 10 differently-named DeviceZoneScopedN zones
// (with increasing durations) on ALL 5 RISCs of a small core grid, then closes. It does NOT drive the X280
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

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    uint32_t gx = 2, gy = 2, n_iters = 50;  // small grid + modest iteration count keep the run quick
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        uint32_t v = (uint32_t)std::strtoul(argv[i + 1], nullptr, 10);
        if (a == "--gx") {
            gx = v;
        } else if (a == "--gy") {
            gy = v;
        } else if (a == "--iters") {
            n_iters = v;
        }
    }

    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    CoreRange cores(CoreCoord{0, 0}, CoreCoord{gx - 1, gy - 1});
    std::map<std::string, std::string> defs{{"N_ITERS", std::to_string(n_iters) + "u"}};
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

    workload.add_program(device_range, std::move(program));
    printf("[perf-debug zones] dispatching %ux%u cores x 5 RISCs x 10 named zones x %u iters ...\n", gx, gy, n_iters);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
    printf("[perf-debug zones] workload done; closing device.\n");
    mesh_device->close();
    return 0;
}
