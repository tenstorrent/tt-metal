// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Each program waits for one wrap of the wall clock's low word (up to 2^32 cycles) on BRISC and NCRISC of every worker
// core and emits profiler zones across it.
constexpr int PROGRAM_COUNT = 3;
constexpr uint32_t ZONE_COUNT = 96;   // Two markers each; fits one RISC's L1 marker buffer.
constexpr uint32_t WRAP_LEAD = 1500;  // Cycles before the wrap at which the zones start.

void RunZonesAcrossWallClockWrap(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores({0, 0}, {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);

    std::map<std::string, std::string> kernel_defines = {
        {"ZONE_COUNT", std::to_string(ZONE_COUNT)}, {"WRAP_LEAD", std::to_string(WRAP_LEAD)}};

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/wall_clock_wrap.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/wall_clock_wrap.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = kernel_defines});

    workload.add_program(device_range, std::move(program));
    for (int i = 0; i < PROGRAM_COUNT; i++) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main() {
    bool pass = true;

    try {
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        RunZonesAcrossWallClockWrap(mesh_device);
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
