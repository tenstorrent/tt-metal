// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

// Phase 1 of the X280<->Tensix experiment: launch a kernel on logical core
// {0,0} that increments a counter at L1 0x80000 forever. The host stays alive
// (the kernel never returns) while the X280 polls the counter through a NoC
// TLB window. L1-only: no DRAM buffers may be touched while Linux owns D5-D7.

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int argc, char** argv) {
    // Logical device id. On multi-chip boxes metal's logical id != PCIe id, so to
    // hit the same physical chip as the X280 (booted on /dev/tenstorrent/N = PCIe
    // id N) pass the logical id that maps to that PCIe id (see UMD "local chip
    // ids/PCIe ids" log line at device open).
    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    Program program = CreateProgram();

    // Launch the counter kernel on EVERY Tensix worker core. Each core increments
    // all 16 u32 of a reserved 64B region (one NoC flit) at L1 0x80000 forever.
    IDevice* dev = mesh_device->get_devices().at(0);
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});

    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "loopback/kernels/counter.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Dump every worker core's NOC0 coordinate so the X280 poller can target them.
    fmt::print(
        "device {} (logical): grid {}x{}, counter (16x u32 @ L1 0x80000) on all cores\n", device_id, grid.x, grid.y);
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            const CoreCoord noc0 = dev->worker_core_from_logical_core(CoreCoord{lx, ly});
            fmt::print("CORE {} {}\n", noc0.x, noc0.y);
        }
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    fmt::print("counter kernel launched; host sleeping (Ctrl-C to stop)\n");
    std::fflush(stdout);

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
    return 0;
}
