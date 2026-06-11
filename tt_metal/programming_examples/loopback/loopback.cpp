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
    constexpr CoreCoord core = {0, 0};

    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "loopback/kernels/counter.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const CoreCoord noc0 = mesh_device->get_devices().at(0)->worker_core_from_logical_core(core);
    fmt::print(
        "device {} (logical): logical (0,0) -> worker core x={} y={}; counter at L1 0x80000\n",
        device_id,
        noc0.x,
        noc0.y);

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
