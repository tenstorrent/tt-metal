// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::vector;
using namespace tt;

void RunTest(IDevice* device) {
    // Set up program
    Program program = Program();
    CoreRange core_range({0, 0}, {5, 5});

    auto l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    // Kernels on brisc + ncrisc that just add two numbers
    KernelHandle brisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_unreserved_base}});
    KernelHandle ncrisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {l1_unreserved_base + 4}});

    // Write runtime args
    auto get_first_arg = [](IDevice* device, CoreCoord& core, uint32_t multiplier) {
        return (uint32_t)device->id() + (uint32_t)core.x * 10 * multiplier;
    };
    auto get_second_arg = [](IDevice* device, CoreCoord& core, uint32_t multiplier) {
        return (uint32_t)core.y * 100 * multiplier;
    };

    for (CoreCoord core : core_range) {
        std::vector<uint32_t> brisc_rt_args = {get_first_arg(device, core, 1), get_second_arg(device, core, 1)};
        std::vector<uint32_t> ncrisc_rt_args = {get_first_arg(device, core, 2), get_second_arg(device, core, 2)};
        SetRuntimeArgs(program, brisc_kid, core, brisc_rt_args);
        SetRuntimeArgs(program, ncrisc_kid, core, ncrisc_rt_args);
    }

    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        // Slow dispatch uses LaunchProgram
        tt::tt_metal::detail::LaunchProgram(device, program);
    } else {
        // Fast Dispatch uses the command queue
        CommandQueue& cq = device->command_queue();
        EnqueueProgram(cq, program, false);
        Finish(cq);
    }

    // Check results
    for (CoreCoord core : core_range) {
        std::vector<uint32_t> brisc_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, sizeof(uint32_t), brisc_result);
        std::vector<uint32_t> ncrisc_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base + 4, sizeof(uint32_t), ncrisc_result);
        uint32_t expected_result = get_first_arg(device, core, 1) + get_second_arg(device, core, 1);
        if (expected_result != brisc_result[0]) {
            log_warning(
                LogTest,
                "Device {}, Core {}, BRISC result was incorrect. Expected {} but got {}",
                device->id(),
                core.str(),
                expected_result,
                brisc_result[0]);
        }
        EXPECT_TRUE(expected_result == brisc_result[0]);
        expected_result = get_first_arg(device, core, 2) + get_second_arg(device, core, 2);
        if (expected_result != ncrisc_result[0]) {
            log_warning(
                LogTest,
                "Device {}, Core {}, NCRISC result was incorrect. Expected {} but got {}",
                device->id(),
                core.str(),
                expected_result,
                ncrisc_result[0]);
        }
        EXPECT_TRUE(expected_result == ncrisc_result[0]);
    }
}

TEST(DispatchStress, TensixRunManyTimes) {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    // Skip fast dispatch until it's supported for remote device.
    if (!slow_dispatch) {
        GTEST_SKIP();
    }
    // Run 500 times to make sure that things work
    for (int idx = 0; idx < 400; idx++) {
        log_info(LogTest, "Running iteration #{}", idx);
        // Need to open/close the device each time in order to reproduce original issue.
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> chip_ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            chip_ids.push_back(id);
        }
        vector<IDevice*> devices_;
        auto reserved_devices_ = tt::tt_metal::detail::CreateDevices(chip_ids);
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }

        // Run the test on each device
        for (IDevice* device : devices_) {
            log_info(LogTest, "Running on device {}", device->id());
            RunTest(device);
        }

        // Close all devices
        tt::tt_metal::detail::CloseDevices(reserved_devices_);
    }
}

}  // namespace tt::tt_metal
