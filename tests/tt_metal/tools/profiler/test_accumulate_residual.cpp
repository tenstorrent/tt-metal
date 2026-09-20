// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <memory>
#include <string>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

int main() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    const CoreCoord core{1, 0};
    const std::map<std::string, std::string> defines{{"LOOP_COUNT", "1"}, {"LOOP_SIZE", "1"}};
    CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/full_buffer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // The single launch leaves its profiler markers below the DRAM-flush
    // threshold. The mid-run read must still parse their L1 identity header.
    ReadMeshDeviceProfilerResults(*mesh_device);
    return mesh_device->close() ? 0 : 1;
}
