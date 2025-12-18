// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

void RunFillUpAllBuffers(const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool fast_dispatch) {
    constexpr CoreCoord core = {0, 0};

    // Mesh workload + device range span the mesh; program encapsulates kernels
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_device_api_debugger/kernels/debug_packets.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = {{"START_DELAY", "0"}}});

    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_device_api_debugger/kernels/debug_packets.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = {{"START_DELAY", "1000"}}});

    workload.add_program(device_range, std::move(program));
    if (fast_dispatch) {
        // Enqueue the same mesh workload multiple times to generate profiler events
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    } else {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main() {
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

    RunFillUpAllBuffers(mesh_device, USE_FAST_DISPATCH);

    return 0;
}
