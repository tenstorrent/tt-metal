// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "single_device_fixture.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace unit_tests::runtime_args {

Program init_compile_and_configure_program(Device *device, const CoreRangeSet &core_range_set) {
    Program program = tt_metal::Program();

    auto add_two_ints_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/riscv_draft/add_two_ints.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    CompileProgram(device, program);
    return std::move(program);
}

bool verify_result(
    Device *device, const Program &program, const std::map<CoreCoord, std::vector<uint32_t>> &core_to_rt_args) {
    bool pass = true;
    auto get_runtime_arg_addr = [](Kernel *kernel) {
        uint32_t result_base = 0;
        switch (kernel->processor()) {
            case tt::RISCV::BRISC: {
                result_base = BRISC_L1_ARG_BASE;
            } break;
            case tt::RISCV::NCRISC: {
                result_base = NCRISC_L1_ARG_BASE;
            } break;
            default: log_assert(false, "Only BRISC and NCRISC have runtime arg support");
        }
        return result_base;
    };

    EXPECT_TRUE(
        program.kernel_ids().size() == 3);  // 2 Blanks get auto-populated even though we added 1 kernel into program
    tt_metal::Kernel *kernel = tt_metal::detail::GetKernel(program, program.kernel_ids().at(0));
    auto processor = kernel->processor();
    auto rt_arg_addr = get_runtime_arg_addr(kernel);

    for (auto kernel_id : program.kernel_ids()) {
        const auto kernel = tt_metal::detail::GetKernel(program, kernel_id);
        auto processor = kernel->processor();
        for (const auto &[logical_core, rt_args] : kernel->runtime_args()) {
            auto expected_rt_args = core_to_rt_args.at(logical_core);
            EXPECT_TRUE(rt_args == expected_rt_args);
            std::vector<uint32_t> written_args;
            tt_metal::detail::ReadFromDeviceL1(
                device, logical_core, rt_arg_addr, rt_args.size() * sizeof(uint32_t), written_args);
            bool got_expected_result = rt_args == written_args;
            EXPECT_TRUE(got_expected_result);
            pass &= got_expected_result;
        }
    }
    return pass;
}

}  // namespace unit_tests::runtime_args

TEST_F(SingleDeviceFixture, LegallyModifyRTArgs) {
    // First run the program with the initial runtime args
    CoreRange first_core_range = {.start = CoreCoord(0, 0), .end = CoreCoord(1, 1)};
    CoreRange second_core_range = {.start = CoreCoord(3, 3), .end = CoreCoord(5, 5)};
    CoreRangeSet core_range_set({first_core_range, second_core_range});
    auto program = unit_tests::runtime_args::init_compile_and_configure_program(this->device_, core_range_set);
    ASSERT_TRUE(
        program.kernel_ids().size() == 3);  // 2 Blanks get auto-populated even though we added 1 kernel into program
    std::vector<uint32_t> initial_runtime_args = {101, 202};
    SetRuntimeArgs(program, program.kernel_ids().at(0), core_range_set, initial_runtime_args);

    std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = initial_runtime_args;
            }
        }
    }
    detail::WriteRuntimeArgsToDevice(this->device_, program);
    ASSERT_TRUE(unit_tests::runtime_args::verify_result(this->device_, program, core_to_rt_args));

    std::vector<uint32_t> second_runtime_args = {303, 606};
    SetRuntimeArgs(program, program.kernel_ids().at(0), first_core_range, second_runtime_args);
    detail::WriteRuntimeArgsToDevice(this->device_, program);
    for (auto x = first_core_range.start.x; x <= first_core_range.end.x; x++) {
        for (auto y = first_core_range.start.y; y <= first_core_range.end.y; y++) {
            CoreCoord logical_core(x, y);
            core_to_rt_args[logical_core] = second_runtime_args;
        }
    }
    EXPECT_TRUE(unit_tests::runtime_args::verify_result(this->device_, program, core_to_rt_args));
}

TEST_F(SingleDeviceFixture, IllegallyModifyRTArgs) {
    // First run the program with the initial runtime args
    CoreRange first_core_range = {.start = CoreCoord(0, 0), .end = CoreCoord(1, 1)};
    CoreRange second_core_range = {.start = CoreCoord(3, 3), .end = CoreCoord(5, 5)};
    CoreRangeSet core_range_set({first_core_range, second_core_range});
    auto program = unit_tests::runtime_args::init_compile_and_configure_program(this->device_, core_range_set);
    ASSERT_TRUE(
        program.kernel_ids().size() == 3);  // 2 Blanks get auto-populated even though we added 1 kernel into program
    std::vector<uint32_t> initial_runtime_args = {101, 202};
    SetRuntimeArgs(program, program.kernel_ids().at(0), core_range_set, initial_runtime_args);

    std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = initial_runtime_args;
            }
        }
    }
    detail::WriteRuntimeArgsToDevice(this->device_, program);
    ASSERT_TRUE(unit_tests::runtime_args::verify_result(this->device_, program, core_to_rt_args));
    std::vector<uint32_t> invalid_runtime_args = {303, 404, 505};
    EXPECT_ANY_THROW(SetRuntimeArgs(program, program.kernel_ids().at(0), first_core_range, invalid_runtime_args));
}
