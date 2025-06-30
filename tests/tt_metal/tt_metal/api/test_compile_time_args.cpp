// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using namespace tt;
using namespace tt::tt_metal;

TEST_F(DeviceFixture, TensixTestTwentyThousandCompileTimeArgs) {
    for (IDevice* device : this->devices_) {
        CoreCoord core = {0, 0};
        Program program;

        const uint32_t write_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

        const std::map<std::string, std::string>& defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}};

        const uint32_t num_compile_time_args = 20000;
        std::vector<uint32_t> compile_time_args(num_compile_time_args);
        std::iota(compile_time_args.begin(), compile_time_args.end(), 0);

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

        const std::vector<uint32_t> compile_time_args_expected{
            std::accumulate(compile_time_args.begin(), compile_time_args.end(), 0u)};

        std::vector<uint32_t> compile_time_args_actual;
        detail::ReadFromDeviceL1(device, core, write_addr, sizeof(uint32_t), compile_time_args_actual);

        ASSERT_EQ(compile_time_args_actual, compile_time_args_expected);
    }
}
