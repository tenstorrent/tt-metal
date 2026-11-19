// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DeviceZoneScopedNIf in the DRAM profiler backend: one program on every worker core whose kernels (DM0, DM1
// and compute) open two conditional zones per iteration, one active and one inactive, both decided by a kernel
// define (a constant expression). test_device_profiler.py::test_conditional_zones reads the device CSV: the
// active zone appears LOOP_COUNT times per RISC with balanced START/END, the inactive one not at all.
// LOOP_COUNT makes the active zone fill most of the ~125-instance buffer, so an inactive zone that still took
// slots would cut the active one short.

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Also in test_device_profiler.py::test_conditional_zones.
constexpr int LOOP_COUNT = 110;
constexpr int LOOP_SIZE = 50;

void RunConditionalZones(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    CoreCoord compute_with_storage_size = mesh_device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);

    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(LOOP_COUNT)},
        {"LOOP_SIZE", std::to_string(LOOP_SIZE)},
        {"ZONE_CT_ON", "1"},
        {"ZONE_CT_OFF", "0"}};

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/conditional_zones_dm.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/conditional_zones_dm.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = kernel_defines});
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/conditional_zones_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{.defines = kernel_defines});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

int main() {
    bool pass = true;

    try {
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        RunConditionalZones(mesh_device);
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
