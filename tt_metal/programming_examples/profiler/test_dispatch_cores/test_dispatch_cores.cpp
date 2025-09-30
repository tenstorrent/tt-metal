// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

void RunCustomCycle(const std::shared_ptr<distributed::MeshDevice>& mesh_device, int loop_count) {
    CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);

    // Mesh workload + device range span the mesh; program encapsulates kernels
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr int loop_size = 50;
    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = kernel_defines});

    std::vector<uint32_t> trisc_kernel_args = {};
    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines});

    // Enqueue mesh workload (non-blocking) and read profiler results
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    ReadMeshDeviceProfilerResults(*mesh_device);
}

int main() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        int loop_count = 2000;
        RunCustomCycle(mesh_device, loop_count);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        fmt::print(stderr, "{}\n", e.what());
        // Capture system call errors that may have returned from driver/kernel
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
