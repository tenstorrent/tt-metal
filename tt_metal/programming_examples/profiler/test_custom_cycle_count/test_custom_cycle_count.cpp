// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using namespace tt;
using namespace tt::tt_metal;

bool RunCustomCycle(const std::shared_ptr<distributed::MeshDevice>& mesh_device, int loop_count) {
    bool pass = true;

    constexpr int loop_size = 50;

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const NodeCoord node{0, 0};

    const KernelSpec::CompilerOptions::Defines defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    const std::string dm_src =
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count.cpp";
    const std::string compute_src =
        "tt_metal/programming_examples/profiler/test_custom_cycle_count/kernels/custom_cycle_count_compute.cpp";

    // The kernel layout is architecture-dependent because num_threads > 1 is only legal on Quasar.
    std::vector<KernelSpec> kernel_specs;
    std::vector<KernelSpecName> wu_kernels;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        // One multi-threaded DM kernel across all six user DMs (DM2-DM7; DM0/DM1 are reserved) plus one
        // four-Neo compute kernel.
        const KernelSpecName DM_KERNEL{"custom_cycle_count_dm"};
        const KernelSpecName COMPUTE_KERNEL{"custom_cycle_count_compute"};
        kernel_specs = {
            KernelSpec{
                .unique_id = DM_KERNEL,
                .source = dm_src,
                .num_threads = 6,
                .compiler_options = {.defines = defines},
                .hw_config = DataMovementHardwareConfig{},
            },
            KernelSpec{
                .unique_id = COMPUTE_KERNEL,
                .source = compute_src,
                .num_threads = 4,
                .compiler_options = {.defines = defines},
                .hw_config = ComputeHardwareConfig{},
            },
        };
        wu_kernels = {DM_KERNEL, COMPUTE_KERNEL};
    } else {
        const KernelSpecName BRISC_KERNEL{"custom_cycle_count_brisc"};
        const KernelSpecName NCRISC_KERNEL{"custom_cycle_count_ncrisc"};
        const KernelSpecName COMPUTE_KERNEL{"custom_cycle_count_compute"};
        kernel_specs = {
            KernelSpec{
                .unique_id = BRISC_KERNEL,
                .source = dm_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config =
                    DataMovementHardwareConfig{
                        .config_1xx =
                            DataMovementHardwareConfig::DataMovement1XXConfig{
                                .processor = DataMovementProcessor::RISCV_0,
                                .noc = NOC::RISCV_0_default,
                            },
                    },
            },
            KernelSpec{
                .unique_id = NCRISC_KERNEL,
                .source = dm_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config =
                    DataMovementHardwareConfig{
                        .config_1xx =
                            DataMovementHardwareConfig::DataMovement1XXConfig{
                                .processor = DataMovementProcessor::RISCV_1,
                                .noc = NOC::RISCV_1_default,
                            },
                    },
            },
            KernelSpec{
                .unique_id = COMPUTE_KERNEL,
                .source = compute_src,
                .num_threads = 1,
                .compiler_options = {.defines = defines},
                .hw_config = ComputeHardwareConfig{},
            },
        };
        wu_kernels = {BRISC_KERNEL, NCRISC_KERNEL, COMPUTE_KERNEL};
    }

    WorkUnitSpec wu{
        .name = "custom_cycle",
        .kernels = wu_kernels,
        .target_nodes = node,
    };
    ProgramSpec spec{
        .name = "custom_cycle_count",
        .kernels = kernel_specs,
        .work_units = {wu},
    };

    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, spec);
    ProgramRunArgs params;  // kernels take no runtime args
    tt_metal::experimental::SetProgramRunArgs(program, params);

    // Enqueue mesh workload (non-blocking) then read profiler results from the mesh device
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    ReadMeshDeviceProfilerResults(*mesh_device);

    return pass;
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
        pass &= RunCustomCycle(mesh_device, loop_count);

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
