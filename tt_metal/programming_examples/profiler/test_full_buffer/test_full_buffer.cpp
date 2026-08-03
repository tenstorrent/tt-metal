// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Local constants for profiler example
constexpr uint32_t DRAM_MARKER_COUNT = 6000;
constexpr uint32_t FULL_L1_MARKER_COUNT = 256;

void RunFillUpAllBuffers(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, int loop_count, bool fast_dispatch) {
    CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    const experimental::NodeRange all_nodes(
        experimental::NodeCoord{0, 0},
        experimental::NodeCoord{compute_with_storage_size.x - 1, compute_with_storage_size.y - 1});

    // Mesh workload + device range span the mesh; program encapsulates kernels
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr int loop_size = 200;
    const experimental::KernelSpec::CompilerOptions::Defines defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    const std::string dm_src = "tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer.cpp";
    const std::string compute_src =
        "tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer_compute.cpp";

    std::vector<experimental::KernelSpec> kernel_specs;
    std::vector<experimental::KernelSpecName> wu_kernels;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        const experimental::KernelSpecName DM_KERNEL{"full_buffer_dm"};
        const experimental::KernelSpecName COMPUTE_KERNEL{"full_buffer_compute"};
        kernel_specs = {
            experimental::KernelSpec{
                .unique_id = DM_KERNEL,
                .source = dm_src,
                .num_threads = 6,
                .compiler_options = {.defines = defines},
                .hw_config = experimental::DataMovementHardwareConfig{experimental::DataMovementGen2Config{}},
            },
            experimental::KernelSpec{
                .unique_id = COMPUTE_KERNEL,
                .source = compute_src,
                .num_threads = 4,
                .compiler_options = {.defines = defines},
                .hw_config = experimental::ComputeGen2Config{},
            },
        };
        wu_kernels = {DM_KERNEL, COMPUTE_KERNEL};
    } else {
        const experimental::KernelSpecName BRISC_KERNEL{"full_buffer_brisc"};
        const experimental::KernelSpecName NCRISC_KERNEL{"full_buffer_ncrisc"};
        const experimental::KernelSpecName COMPUTE_KERNEL{"full_buffer_compute"};
        kernel_specs = {
            experimental::KernelSpec{
                .unique_id = BRISC_KERNEL,
                .source = dm_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config = experimental::DataMovementHardwareConfig{experimental::DataMovementGen1Config{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}},
            },
            experimental::KernelSpec{
                .unique_id = NCRISC_KERNEL,
                .source = dm_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config = experimental::DataMovementHardwareConfig{experimental::DataMovementGen1Config{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}},
            },
            experimental::KernelSpec{
                .unique_id = COMPUTE_KERNEL,
                .source = compute_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config = experimental::ComputeGen1Config{},
            },
        };
        wu_kernels = {BRISC_KERNEL, NCRISC_KERNEL, COMPUTE_KERNEL};
    }

    experimental::WorkUnitSpec wu{
        .name = "full_buffer",
        .kernels = wu_kernels,
        .target_nodes = all_nodes,
    };
    experimental::ProgramSpec spec{
        .name = "full_buffer",
        .kernels = kernel_specs,
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs params;  // kernels take no runtime args
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    // One enqueue is enough to fill the L1 profiler buffer on Quasar
    if (fast_dispatch && mesh_device->arch() != tt::ARCH::QUASAR) {
        for (int i = 0; i < DRAM_MARKER_COUNT / FULL_L1_MARKER_COUNT; i++) {
            // Enqueue the same mesh workload multiple times to generate profiler traffic
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        }
    } else {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

        constexpr int device_loop_count = 150;

        RunFillUpAllBuffers(mesh_device, device_loop_count, USE_FAST_DISPATCH);
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
