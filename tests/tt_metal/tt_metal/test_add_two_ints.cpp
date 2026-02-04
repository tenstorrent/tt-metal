// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cstdint>
#include <array>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

////////////////////////////////////////////////////////////////////////////
// Runs the add_two_ints kernel on BRISC to add two ints in L1
// Result is read from L1
////////////////////////////////////////////////////////////////////////////
TEST_F(MeshDeviceSingleCardFixture, AddTwoInts) {
    IDevice* dev = devices_[0]->get_devices()[0];
    uint32_t l1_unreserved_base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);

    Program program = CreateProgram();
    CoreCoord core = {0, 0};
    constexpr std::array<uint32_t, 2> first_runtime_args = {101, 202};
    constexpr std::array<uint32_t, 2> second_runtime_args = {303, 606};

    KernelHandle add_two_ints_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_unreserved_base}});

    // First run
    SetRuntimeArgs(program, add_two_ints_kernel, core, first_runtime_args);
    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> first_kernel_result;
    detail::ReadFromDeviceL1(dev, core, l1_unreserved_base, sizeof(int), first_kernel_result);
    log_info(LogVerif, "first kernel result = {}", first_kernel_result[0]);

    // Second run with updated args
    SetRuntimeArgs(program, add_two_ints_kernel, core, second_runtime_args);
    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> second_kernel_result;
    detail::ReadFromDeviceL1(dev, core, l1_unreserved_base, sizeof(int), second_kernel_result);
    log_info(LogVerif, "second kernel result = {}", second_kernel_result[0]);

    // Validation
    uint32_t first_expected_result = first_runtime_args[0] + first_runtime_args[1];
    uint32_t second_expected_result = second_runtime_args[0] + second_runtime_args[1];
    log_info(
        LogVerif,
        "first expected result = {} second expected result = {}",
        first_expected_result,
        second_expected_result);

    EXPECT_EQ(first_kernel_result[0], first_expected_result);
    EXPECT_EQ(second_kernel_result[0], second_expected_result);
}
