// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <string>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt_metal = tt::tt_metal;

int main() {
    int device_id = 0;
    auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);

    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", "100"},
        {"LOOP_SIZE", "100"},
    };
    auto program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/tensix/kernels/compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .defines = kernel_defines,
        });

    tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt_metal::distributed::MeshCoordinateRange(device->shape()), std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, true);
    tt_metal::ReadMeshDeviceProfilerResults(*device);
    device->close();
}
