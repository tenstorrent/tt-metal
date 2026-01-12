// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    using namespace tt;
    using namespace tt::tt_metal;
    std::cout << "Test started" << std::endl;

    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        std::cerr
            << "ERROR: This test can only be run using a simulator. Please set Environment Variable TT_METAL_SIMULATOR"
            << std::endl;
        std::cerr << "ERROR: with a valid simulator path" << std::endl;
        return 1;
    }
    env_var = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (env_var == nullptr) {
        std::cerr << "ERROR: This test can only be run in slow dispatch mode. Please set Environment Variable "
                     "TT_METAL_SLOW_DISPATCH_MODE"
                  << std::endl;
        std::cerr << "ERROR: using export TT_METAL_SLOW_DISPATCH_MODE=1" << std::endl;
        return 1;
    }

    // Initialize mesh device (1x1), command queue, workload, device range, and program.
    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle kernel_0 = experimental::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core,
        experimental::QuasarDataMovementConfig{
            .num_processors_per_cluster = 8, .compile_args = {MEM_L1_UNCACHED_BASE}});

    // KernelHandle kernel_1 = experimental::CreateKernel(
    //     program,
    //     "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
    //     core,
    //     experimental::QuasarDataMovementConfig{
    //         .num_processors_per_cluster = 1, .compile_args = {MEM_L1_UNCACHED_BASE + 4}});

    SetRuntimeArgs(program, kernel_0, core, {100, 200});
    // SetRuntimeArgs(program, kernel_1, core, {300, 400});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> kernel_0_result{0};
    // std::vector<uint32_t> kernel_1_result{0};
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core, MEM_L1_UNCACHED_BASE, sizeof(int), kernel_0_result);
    // tt_metal::detail::ReadFromDeviceL1(
    //     mesh_device->get_devices()[0], core, MEM_L1_UNCACHED_BASE + 4, sizeof(int), kernel_1_result);

    mesh_device->close();

    // if (kernel_0_result[0] == 300 && kernel_1_result[0] == 700) {
    //     std::cout << "Test passed!" << std::endl;
    //     return 0;
    // } else if (kernel_0_result[0] != 300) {
    //     std::cout << "Test failed! Got the value " << kernel_0_result[0] << " instead of " << 300 << std::endl;
    //     return 1;
    // } else if (kernel_1_result[0] != 700) {
    //     std::cout << "Test failed! Got the value " << kernel_1_result[0] << " instead of " << 700 << std::endl;
    //     return 1;
    // }
    if (kernel_0_result[0] == 300) {
        std::cout << "Test passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test failed! Got the value " << kernel_0_result[0] << " instead of " << 300 << std::endl;
        return 1;
    }
}
