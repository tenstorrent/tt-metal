// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include "hostdevcommon/profiler_common.h"

using namespace tt;

void RunFillUpAllBuffers(tt_metal::IDevice* device, int loop_count, bool fast_dispatch) {
    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores(start_core, end_core);
    auto eth_cores = device->get_active_ethernet_cores(true);

    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr int loop_size = 200;
    constexpr int enqueue_times = 2;
    std::map<std::string, std::string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)}, {"LOOP_SIZE", std::to_string(loop_size)}};

    tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_timestamped_events/kernels/timestamped_events.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = kernel_defines});
    tt_metal::KernelHandle ncrisc_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_timestamped_events/kernels/timestamped_events.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = kernel_defines});
    std::vector<uint32_t> trisc_kernel_args = {};
    tt_metal::KernelHandle trisc_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_timestamped_events/kernels/timestamped_events_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines});

    for (auto core : eth_cores) {
        auto eth_reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_timestamped_events/kernels/timestamped_events.cpp",
            (CoreCoord){core.x, core.y},
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .defines = kernel_defines});
    }

    if (fast_dispatch) {
        for (int i = 0; i < enqueue_times; i++) {
            program.set_runtime_id(i);
            EnqueueProgram(device->command_queue(), program, false);
        }
    } else {
        tt_metal::detail::LaunchProgram(device, program);
    }
}

int main() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;

        constexpr int device_loop_count = 150;

        RunFillUpAllBuffers(device, device_loop_count, USE_FAST_DISPATCH);
        tt_metal::detail::DumpDeviceProfileResults(device);

        pass &= tt_metal::CloseDevice(device);

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
