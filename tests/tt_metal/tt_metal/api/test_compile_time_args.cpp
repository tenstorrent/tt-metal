// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device.hpp"
#include "device_fixture.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

TEST_F(DeviceFixture, TensixTestFourHundredCompileTimeArgs) {
    // This test will hang/assert if there is a failure
    for (IDevice* device : this->devices_) {
        CoreCoord core = {0, 0};
        Program program;

        const uint32_t num_compile_time_args = 400;
        std::vector<uint32_t> compile_time_args;
        for (uint32_t i = 0; i < num_compile_time_args; i++) {
            compile_time_args.push_back(i);
        }

        const std::map<string, string>& defines = {{"NUM_COMPILE_TIME_ARGS", std::to_string(num_compile_time_args)}};

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/compile_time_args_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args,
                .defines = defines});
        this->RunProgram(device, program);
    }
}

TEST_F(DeviceFixture, TensixTestZeroCompileTimeArgs) {
    // This test will hang/assert if there is a failure
    for (IDevice* device : this->devices_) {
        CoreCoord core = {0, 0};
        Program program;

        const std::map<string, string>& defines = {{"NUM_COMPILE_TIME_ARGS", "0"}};

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/compile_time_args_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
        this->RunProgram(device, program);
    }
}
