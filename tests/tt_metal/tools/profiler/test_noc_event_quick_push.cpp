// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Reproducer for the device-profiler NoC-event capture of a long kernel: one launch on a few cores, BRISC and NCRISC
// each record LOOP_COUNT NoC reads, which makes the L1 marker ring quick-push to DRAM many times mid-kernel.
// Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 and a program support count large enough
// that the per-RISC DRAM region does not overflow (LOOP_COUNT * 16 B per RISC, 48 B per support count unit).
// NOC_EVENT_QUICK_PUSH_LOOP_COUNT overrides the loop count.
int main() {
    bool pass = true;

    try {
        uint32_t loop_count = 4000;
        if (const char* env = std::getenv("NOC_EVENT_QUICK_PUSH_LOOP_COUNT")) {
            loop_count = static_cast<uint32_t>(std::atoi(env));
        }
        constexpr uint32_t page_size = 64;

        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        CoreRange cores({0, 0}, {1, 1});

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr uint32_t cb_id = 0;
        CircularBufferConfig cb_config =
            CircularBufferConfig(2 * page_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, page_size);
        CreateCircularBuffer(program, cores, cb_config);

        std::map<std::string, std::string> kernel_defines = {{"LOOP_COUNT", std::to_string(loop_count)}};
        const std::string kernel_path = "tests/tt_metal/tools/profiler/kernels/noc_event_quick_push.cpp";
        KernelHandle brisc_kernel = CreateKernel(
            program,
            kernel_path,
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .defines = kernel_defines});
        KernelHandle ncrisc_kernel = CreateKernel(
            program,
            kernel_path,
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .defines = kernel_defines});
        SetRuntimeArgs(program, brisc_kernel, cores, {page_size});
        SetRuntimeArgs(program, ncrisc_kernel, cores, {page_size});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

        ReadMeshDeviceProfilerResults(*mesh_device);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
