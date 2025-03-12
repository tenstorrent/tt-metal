// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device.hpp"
#include "device_fixture.hpp"
#include "hal.hpp"
#include <string>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

TEST_F(DeviceFixture, TensixTestSixtyThousandCompileTimeArgs) {
    for (IDevice* device : this->devices_) {
        CoreCoord core = {0, 0};
        Program program;

        const uint32_t write_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED);

        const std::map<string, string>& defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}};

        const uint32_t num_compile_time_args = 60000;
        std::vector<uint32_t> compile_time_args;
        for (uint32_t i = 0; i < num_compile_time_args; i++) {
            compile_time_args.push_back(1);
        }

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

        const std::vector<uint32_t> compile_time_args_expected{num_compile_time_args};

        std::vector<uint32_t> compile_time_args_actual;
        detail::ReadFromDeviceL1(device, core, write_addr, sizeof(uint32_t), compile_time_args_actual);

        ASSERT_EQ(compile_time_args_actual, compile_time_args_expected);
    }
}
