// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

////////////////////////////////////////////////////////////////////////////
// Runs the add_two_ints kernel on BRISC to add two ints in L1
// Result is read from L1
////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        uint32_t l1_unreserved_base = device->get_base_allocator_addr(tt_metal::HalMemType::L1);


        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord core = {0, 0};
        std::vector<uint32_t> first_runtime_args = {101, 202};
        std::vector<uint32_t> second_runtime_args = {303, 606};

        tt_metal::KernelHandle add_two_ints_kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp", core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {l1_unreserved_base}
            });

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, first_runtime_args);


        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> first_kernel_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, sizeof(int), first_kernel_result);
        log_info(LogVerif, "first kernel result = {}", first_kernel_result[0]);

        ////////////////////////////////////////////////////////////////////////////
        //                  Update Runtime Args and Re-run Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, second_runtime_args);
        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> second_kernel_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, sizeof(int), second_kernel_result);
        log_info(LogVerif, "second kernel result = {}", second_kernel_result[0]);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        uint32_t first_expected_result = first_runtime_args[0] + first_runtime_args[1];
        uint32_t second_expected_result = second_runtime_args[0] + second_runtime_args[1];
        log_info(LogVerif, "first expected result = {} second expected result = {}", first_expected_result, second_expected_result);
        pass = first_kernel_result[0] == first_expected_result;
        pass = second_kernel_result[0] == second_expected_result;

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

    TT_FATAL(pass, "Error");


    return 0;
}
