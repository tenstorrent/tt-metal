// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <string>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/observability/profiler/device_profiler.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <tt-metalium/execution/distributed.hpp>

namespace tt_metal = tt::tt_metal;

int main() {
    int device_id = 0;
    auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    tt::tt_metal::CoreCoord start_core = {0, 0};
    tt::tt_metal::CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    tt::tt_metal::CoreRange all_cores(start_core, end_core);

    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", "100"},
        {"LOOP_SIZE", "100"},
    };
    auto program = tt_metal::CreateProgram(*device);
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
