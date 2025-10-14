// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt-metalium/kernel_types.hpp"
#include "tt_stl/assert.hpp"

using namespace tt::tt_metal;

constexpr std::string_view null_kernel_dm =
    R"(
void kernel_main() {
}
)";

constexpr std::string_view null_kernel_compute =
    R"(
#include "compute_kernel_api.h"
namespace NAMESPACE {
void MAIN {
}
}
)";

int main() {
    // Emulate we are executing and using kernels outside of TT_METAL_HOME
    chdir("/tmp");

    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Generate the kernels
    {
        std::ofstream null_kernel_compute_out("null_kernel_compute.cpp");
        std::ofstream null_kernel_dm_out("null_kernel_dm.cpp");
        TT_ASSERT(null_kernel_compute_out.is_open(), "Failed to open file for writing");
        TT_ASSERT(null_kernel_dm_out.is_open(), "Failed to open file for writing");
        null_kernel_compute_out << null_kernel_compute;
        null_kernel_dm_out << null_kernel_dm;
    }

    constexpr CoreCoord core = {0, 0};

    CreateKernel(
        program,
        "./null_kernel_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(program, "./null_kernel_compute.cpp", core, ComputeConfig{});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    // Close the device
    if (!mesh_device->close()) {
        return 1;
    }

    return 0;
}
