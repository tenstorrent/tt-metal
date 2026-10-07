// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <utility>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

int main() {
    TT_FATAL(std::getenv("TT_METAL_DEVICE_PROFILER") != nullptr, "Enable TT_METAL_DEVICE_PROFILER=1");
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/remote_compile_zones.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);
    ReadMeshDeviceProfilerResults(*mesh_device);
    return mesh_device->close() ? 0 : 1;
}
