// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"



using namespace tt;
using namespace tt::tt_metal;

namespace unit_tests::runtime_args {

enum class KernelType {
    DATA_MOVEMENT = 0,
    COMPUTE = 1,
};


Program initialize_program_data_movement(Device *device, const CoreRangeSet &core_range_set) {
    Program program = tt_metal::CreateProgram();

    auto add_two_ints_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    detail::CompileProgram(device, program);
    return std::move(program);
}



Program initialize_program_compute(Device *device, const CoreRangeSet &core_range_set) {
    Program program = tt_metal::CreateProgram();

    std::vector<uint32_t> compute_args = {0};  // dummy
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/increment_runtime_arg.cpp",
        core_range_set,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_args});



    return std::move(program);
}


bool verify_result_data_movement(
    Device *device, const Program &program, const std::map<CoreCoord, std::vector<uint32_t>> &core_to_rt_args) {
    bool pass = true;
    auto get_runtime_arg_addr = [](std::shared_ptr<Kernel> kernel) {
        uint32_t arg_base = 0;
        switch (kernel->processor()) {
            case tt::RISCV::BRISC: {
                arg_base = BRISC_L1_ARG_BASE;
            } break;
            case tt::RISCV::NCRISC: {
                arg_base = NCRISC_L1_ARG_BASE;
            } break;
            default: TT_THROW("Only BRISC and NCRISC have runtime arg support");
        }
        return arg_base;
    };

    EXPECT_TRUE(
        program.num_kernels() == 1);
    auto kernel = tt_metal::detail::GetKernel(program, 0);
    auto processor = kernel->processor();
    auto rt_arg_addr = get_runtime_arg_addr(kernel);

    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const auto kernel = tt_metal::detail::GetKernel(program, kernel_id);
        auto processor = kernel->processor();
        for (const auto &logical_core : kernel->cores_with_runtime_args()) {
            auto expected_rt_args = core_to_rt_args.at(logical_core);
            auto rt_args = kernel->runtime_args(logical_core);
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

TEST_F(DeviceFixture, LegallyModifyRTArgsDataMovement) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set({first_core_range, second_core_range});
        auto program =
            unit_tests::runtime_args::initialize_program_data_movement(this->devices_.at(id), core_range_set);
        ASSERT_TRUE(
            program.num_kernels() ==
            1);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        ASSERT_TRUE(
            unit_tests::runtime_args::verify_result_data_movement(this->devices_.at(id), program, core_to_rt_args));

        std::vector<uint32_t> second_runtime_args = {303, 606};
        SetRuntimeArgs(program, 0, first_core_range, second_runtime_args);
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        for (auto x = first_core_range.start.x; x <= first_core_range.end.x; x++) {
            for (auto y = first_core_range.start.y; y <= first_core_range.end.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = second_runtime_args;
            }
        }
        EXPECT_TRUE(
            unit_tests::runtime_args::verify_result_data_movement(this->devices_.at(id), program, core_to_rt_args));
    }
}

bool verify_result_compute(
    Device *device, const Program &program,
        const std::map<CoreCoord, std::vector<uint32_t>> &core_to_rt_args,
        KernelType kern_type = KernelType::DATA_MOVEMENT,
        uint32_t buffer_addr = 0) {
    bool pass = true;
    auto get_runtime_arg_addr = [](std::shared_ptr<Kernel> kernel) {
        uint32_t result_base = 0;
        switch (kernel->processor()) {
            case tt::RISCV::BRISC: {
                result_base = BRISC_L1_ARG_BASE;
            } break;
            case tt::RISCV::NCRISC: {
                result_base = NCRISC_L1_ARG_BASE;
            } break;
            case tt::RISCV::COMPUTE: {
                result_base = TRISC_L1_ARG_BASE;
            } break;
            default: TT_THROW("Unknown processor");
        }
        return result_base;
    };


    EXPECT_TRUE(
        program.num_kernels() == 1);
    auto kernel = tt_metal::detail::GetKernel(program, 0);
    auto processor = kernel->processor();
    auto rt_arg_addr = get_runtime_arg_addr(kernel);
    auto rt_result_addr = get_runtime_arg_addr(kernel);

    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const auto kernel = tt_metal::detail::GetKernel(program, kernel_id);
        auto processor = kernel->processor();
        for (const auto &logical_core : kernel->cores_with_runtime_args()) {
            auto expected_rt_args = core_to_rt_args.at(logical_core);
            auto rt_args = kernel->runtime_args(logical_core);
            EXPECT_TRUE(rt_args == expected_rt_args);
            std::vector<uint32_t> written_args;
            tt_metal::detail::ReadFromDeviceL1(
                device, logical_core, rt_arg_addr, rt_args.size() * sizeof(uint32_t), written_args);

            std::vector<uint32_t> increments = {87, 216};
            for(int i=0; i<rt_args.size(); i++){
                bool got_expected_result;
                got_expected_result = written_args[i] == (rt_args[i] + increments[i]);
                pass &= got_expected_result;
                EXPECT_TRUE(got_expected_result);
            }

        }
    }
    return pass;
}

TEST_F(DeviceFixture, LegallyModifyRTArgsCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set({first_core_range, second_core_range});
        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_result_compute(
            this->devices_.at(id), program, core_to_rt_args, KernelType::COMPUTE));
    }
}

TEST_F(DeviceFixture, IllegallyModifyRTArgs) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set({first_core_range, second_core_range});
        auto program =
            unit_tests::runtime_args::initialize_program_data_movement(this->devices_.at(id), core_range_set);
        ASSERT_TRUE(
            program.num_kernels() ==
            1);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        ASSERT_TRUE(
            unit_tests::runtime_args::verify_result_data_movement(this->devices_.at(id), program, core_to_rt_args));
        std::vector<uint32_t> invalid_runtime_args = {303, 404, 505};
        EXPECT_ANY_THROW(SetRuntimeArgs(program, 0, first_core_range, invalid_runtime_args));
    }
}

}  // namespace unit_tests::runtime_args
