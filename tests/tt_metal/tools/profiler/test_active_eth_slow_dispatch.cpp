// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device profiler coverage for kernels on active ethernet cores. Fast dispatch to active ethernet has been removed,
// so this test only runs under slow dispatch and programs nothing but the active ethernet cores of device 0.

#include <cstdint>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/kernels/kernel.hpp"
#include "distributed/mesh_device_impl.hpp"

using namespace tt;
using namespace tt::tt_metal;

void RunActiveEthKernels(const std::shared_ptr<distributed::MeshDevice>& mesh_device, int loop_count) {
    auto* device = mesh_device->impl().get_devices()[0];
    auto eth_cores = device->get_active_ethernet_cores(true);
    if (eth_cores.empty()) {
        fmt::print("No active ethernet cores on device {}, nothing to profile\n", device->id());
        return;
    }

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);

    constexpr int loop_size = 200;
    constexpr int enqueue_times = 2;
    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    for (auto core : eth_cores) {
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/timestamped_events.cpp",
            (CoreCoord){core.x, core.y},
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .defines = kernel_defines});
    }

    workload.add_program(device_range, std::move(program));
    for (int i = 0; i < enqueue_times; i++) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    }
}

int main() {
    bool pass = true;

    try {
        TT_FATAL(
            std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr,
            "Active ethernet kernels require slow dispatch; set TT_METAL_SLOW_DISPATCH_MODE=1");

        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr int device_loop_count = 150;

        RunActiveEthKernels(mesh_device, device_loop_count);
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
