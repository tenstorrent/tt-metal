// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t OP_COUNT = 1400;

uint32_t program_runtime_id = 1;

void RunCustomCycle(const std::shared_ptr<distributed::MeshDevice>& mesh_device, int num_ops) {
    const CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    const CoreCoord start_core = {0, 0};
    const CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    const CoreRange all_cores(start_core, end_core);

    for (uint32_t i = 0; i < num_ops; ++i) {
        // Mesh workload + device range span the mesh; program encapsulates kernels
        distributed::MeshWorkload workload;
        const distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        tt_metal::Program program = tt_metal::CreateProgram();

        if (i % 5 == 0) {
            tt_metal::CreateKernel(
                program,
                "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op_compute.cpp",
                all_cores,
                tt_metal::ComputeConfig{.compile_args = {}});
        }
        tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        program.set_runtime_id(program_runtime_id++);
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        constexpr ChipId device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        RunCustomCycle(mesh_device, OP_COUNT);
        ReadMeshDeviceProfilerResults(*mesh_device);

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
