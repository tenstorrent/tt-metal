// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The Metal process: runs kernels/publish_telemetry.cpp on a 2x2 grid of device 0, which sends its telemetry samples to
// the external process. Start metal_example_telemetry_external_process first, then run this with the streaming
// profiler on:
//
//   TT_METAL_STREAMING_PROFILER=1 metal_example_telemetry_metal_process [seconds] [samples per second]

#include <cstdint>
#include <string>

#include <fmt/format.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt::tt_metal;

int main(int argc, char** argv) {
    const uint32_t seconds = argc > 1 ? std::stoul(argv[1]) : 5;
    const uint32_t rate = argc > 2 ? std::stoul(argv[2]) : 10;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    // The kernel paces itself on the core's clock, which counts AICLK cycles.
    const uint32_t interval_cycles = static_cast<uint32_t>(mesh_device->get_clock_rate_mhz()) * 1'000'000 / rate;

    Program program = CreateProgram();
    const CoreRange cores(CoreCoord{0, 0}, CoreCoord{1, 1});
    const KernelHandle kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "profiler/streaming_profiler_telemetry/kernels/publish_telemetry.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    uint32_t bytes_per_sample = 4096;
    for (const CoreCoord& core : cores) {
        SetRuntimeArgs(program, kernel, core, {seconds * rate, interval_cycles, bytes_per_sample});
        bytes_per_sample *= 2;
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    fmt::print("publishing telemetry {} times per second for {} s from {} cores\n", rate, seconds, cores.size());
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
    mesh_device->close();
    return 0;
}
