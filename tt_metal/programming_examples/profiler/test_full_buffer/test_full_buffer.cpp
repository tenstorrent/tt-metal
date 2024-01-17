// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;

bool RunCustomCycle(tt_metal::Device *device, int loop_count, int run_count, int fastDispatch)
{
    bool pass = true;

    CoreCoord compute_with_storage_size = device->compute_with_storage_grid_size();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
    CoreRange all_cores{.start=start_core, .end=end_core};

    tt_metal::Program program = tt_metal::CreateProgram();

    constexpr int loop_size = 200;
    std::map<string, string> kernel_defines = {
        {"LOOP_COUNT", std::to_string(loop_count)},
        {"LOOP_SIZE", std::to_string(loop_size)}
    };

    if (run_count % 3 | !(run_count % 4) | (fastDispatch > 1))
    {
        tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
            program, "tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = kernel_defines});
    }

    if (run_count % 4 | !(run_count % 3) | (fastDispatch > 1))
    {
        tt_metal::KernelHandle ncrisc_kernel = tt_metal::CreateKernel(
            program, "tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines = kernel_defines});
    }

    if ((run_count % 5) | (fastDispatch > 1))
    {
        vector<uint32_t> trisc_kernel_args = {};
        tt_metal::KernelHandle trisc_kernel = tt_metal::CreateKernel(
            program, "tt_metal/programming_examples/profiler/test_full_buffer/kernels/full_buffer_compute.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = trisc_kernel_args, .defines = kernel_defines});
    }

    for (int i = 0; i < fastDispatch; i++)
    {
        EnqueueProgram(tt_metal::detail::GetCommandQueue(device), program, false);
    }

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);

        constexpr int device_loop_count = 10;
        constexpr int host_loop_count = 60;

        for (int i = 0; i < host_loop_count; i ++)
        {
            pass &= RunCustomCycle(device, device_loop_count, i, 1);
        }

        Finish(tt_metal::detail::GetCommandQueue(device));
        tt_metal::detail::DumpDeviceProfileResults(device);

        tt_metal::detail::InitDeviceProfiler(device);

        pass &= RunCustomCycle(device, device_loop_count, 0, host_loop_count);

        Finish(tt_metal::detail::GetCommandQueue(device));
        tt_metal::detail::DumpDeviceProfileResults(device);
        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
