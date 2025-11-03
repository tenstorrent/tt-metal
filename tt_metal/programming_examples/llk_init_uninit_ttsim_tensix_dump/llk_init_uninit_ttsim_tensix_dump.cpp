// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// #include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
// #include "tt-metalium/buffer.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    // Ensure cores are selected for DPRINT so we can see the output of the TTSIM_TENSIX_DUMP debug feature.
    char* dprint_cores_selected = std::getenv("TT_METAL_DPRINT_CORES");
    if (dprint_cores_selected == nullptr) {
        fmt::print(
            "ERROR: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the "
            "TTSIM_TENSIX_DUMP debug feature.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
        return -1;
    }

    // Ensure the TTSIM is enabled so the TTSIM_TENSIX_DUMP debug feature is enabled.
    char* ttsim_enabled = std::getenv("TT_METAL_SIMULATOR");
    if (ttsim_enabled == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_SIMULATOR to the path to your libttsim.so file to "
            "enable the TTSIM_TENSIX_DUMP debug feature.\n");
        fmt::print("WARNING: For example, export export TT_METAL_SIMULATOR=~/sim/libttsim.so\n");
    }

    // A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a
    // single object, maintaining uniform memory and runtime state across all physical devices. A UnitMesh is a 1x1
    // MeshDevice that allows users to interface with a single physical device.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // In Metalium, submitting operations to the device is done through a command queue. This includes
    // uploading/downloading data to/from the device, and executing programs.
    // A MeshCommandQueue is a software concept that allows developers to submit operations to a MeshDevice.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // A MeshWorkload is a collection of programs that are executed on a MeshDevice.
    // The specific physical devices that the workload is executed on are determined by the MeshCoordinateRange.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // A Program contains kernels that perform computations or data movement.
    Program program = CreateProgram();
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr CoreCoord core = {0, 0};

    // Create the kernel that will be used to demo the TTSIM_TENSIX_DUMP debug feature.
    CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "llk_init_uninit_ttsim_tensix_dump/kernels/compute_llk_ttsim_tensix_dump.cpp",
        core,
        ComputeConfig{});

    // Add the program to the workload and enqueue it for execution on the MeshDevice.
    // Setting blocking=false returns immediately; commands on the queue execute in FIFO order.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device->close();
}
