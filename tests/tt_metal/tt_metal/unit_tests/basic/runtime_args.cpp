// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"

#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"


using namespace tt;
using namespace tt::tt_metal;

namespace unit_tests::runtime_args {

enum class KernelType {
    DATA_MOVEMENT = 0,
    COMPUTE = 1,
};

uint32_t get_runtime_arg_addr(uint32_t l1_unreserved_base, tt::RISCV processor, bool is_common) {
    uint32_t result_base = 0;

    // Spread results out a bit, overly generous
    constexpr uint32_t runtime_args_space = 1024 * sizeof(uint32_t);

    switch (processor) {
        case tt::RISCV::BRISC: {
            result_base = l1_unreserved_base;
        } break;
        case tt::RISCV::NCRISC: {
            result_base = l1_unreserved_base + 1 * runtime_args_space;
        } break;
        case tt::RISCV::COMPUTE: {
            result_base = l1_unreserved_base + 2 * runtime_args_space;
        } break;
        default: TT_THROW("Unknown processor");
    }

    uint32_t offset = is_common ? 3 * runtime_args_space : 0;
    return result_base + offset;
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

Program initialize_program_data_movement_rta(Device *device, const CoreRangeSet &core_range_set, uint32_t num_unique_rt_args,
                                             bool common_rtas = false) {
    Program program = tt_metal::CreateProgram();

    uint32_t rta_base_dm = get_runtime_arg_addr(device->get_base_allocator_addr(HalMemType::L1), tt::RISCV::BRISC, common_rtas);
    std::map<string, string> dm_defines = {{"DATA_MOVEMENT", "1"},
                                           {"NUM_RUNTIME_ARGS", std::to_string(num_unique_rt_args)},
                                           {"RESULTS_ADDR", std::to_string(rta_base_dm)}};
    if (common_rtas) {
        dm_defines["COMMON_RUNTIME_ARGS"] = "1";
    }

    auto kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        core_range_set,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = dm_defines});

    detail::CompileProgram(device, program);
    return std::move(program);
}

Program initialize_program_compute(Device *device, const CoreRangeSet &core_range_set, uint32_t num_unique_rt_args, uint32_t num_common_rt_args) {
    Program program = tt_metal::CreateProgram();

    // Tell kernel how many unique and common RT args to expect. Will increment each.
    uint32_t rta_base_compute = get_runtime_arg_addr(device->get_base_allocator_addr(HalMemType::L1), tt::RISCV::COMPUTE, false);
    uint32_t common_rta_base_compute = get_runtime_arg_addr(device->get_base_allocator_addr(HalMemType::L1), tt::RISCV::COMPUTE, true);
    std::vector<uint32_t> compile_args = {num_unique_rt_args, num_common_rt_args, rta_base_compute, common_rta_base_compute};
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/increment_runtime_arg.cpp",
        core_range_set,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compile_args});

    return std::move(program);
}

// Verify the runtime args for a single core (apply optional non-zero increment amounts to values written to match compute kernel)
bool verify_core_rt_args(Device *device, bool is_common, CoreCoord core, uint32_t base_addr, const std::vector<uint32_t> &written_args, const uint32_t incr_val) {
    bool pass = true;

    std::vector<uint32_t> observed_args;
    tt_metal::detail::ReadFromDeviceL1(device, core, base_addr, written_args.size() * sizeof(uint32_t), observed_args);

    for (size_t i = 0; i < written_args.size(); i++) {
        uint32_t expected_result = written_args.at(i) + incr_val;
        bool got_expected_result = observed_args.at(i) == expected_result;
        log_debug(tt::LogTest, "Validating {} Args. Core: {} at addr: 0x{:x} idx: {} - Expected: {} Observed: {}",
            is_common ? "Common" : "Unique", core.str(), base_addr, i, expected_result, observed_args[i]);
        pass &= got_expected_result;
        EXPECT_TRUE(got_expected_result);
    }
    return pass;
}

// Iterate over all cores unique and common runtime args, and verify they match expected values.
bool verify_results(
    bool are_args_incremented, Device *device, const Program &program, const std::map<CoreCoord, std::vector<uint32_t>> &core_to_rt_args, const std::vector<uint32_t> &common_rt_args = {}) {

    bool pass = true;
    EXPECT_TRUE(program.num_kernels() == 1);

    // These increment amounts model what is done by compute kernel in this test.
    uint32_t unique_arg_incr_val = are_args_incremented ? 10 : 0;
    uint32_t common_arg_incr_val = are_args_incremented ? 100 : 0;

    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const auto kernel = tt_metal::detail::GetKernel(program, kernel_id);
        auto rt_args_base_addr = get_runtime_arg_addr(device->get_base_allocator_addr(HalMemType::L1), kernel->processor(), false);

        // Verify Unique RT Args (per core)
        for (const auto &logical_core : kernel->cores_with_runtime_args()) {
            auto expected_rt_args = core_to_rt_args.at(logical_core);
            auto rt_args = kernel->runtime_args(logical_core);
            EXPECT_EQ(rt_args, expected_rt_args);

            pass &= verify_core_rt_args(device, false, logical_core, rt_args_base_addr, expected_rt_args, unique_arg_incr_val);
            auto rt_args_size_bytes = rt_args.size() * sizeof(uint32_t);
        }

        // Verify common RT Args (same for all cores) if they exist.
        if (common_rt_args.size() > 0) {
            auto common_rt_args_base_addr = get_runtime_arg_addr(device->get_base_allocator_addr(HalMemType::L1), kernel->processor(), true);

            for (auto &core_range : kernel->logical_coreranges()) {
                for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                    for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                        CoreCoord logical_core({x, y});
                        auto rt_args = kernel->common_runtime_args();
                        EXPECT_EQ(rt_args, common_rt_args);
                        pass &= verify_core_rt_args(device, true, logical_core, common_rt_args_base_addr, common_rt_args, common_arg_incr_val);
                    }
                }
            }
        }
    }

    return pass;
}

// Write unique and common runtime args to device and readback to verify written correctly.
TEST_F(DeviceFixture, LegallyModifyRTArgsDataMovement) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        auto program = unit_tests::runtime_args::initialize_program_data_movement_rta(this->devices_.at(id), core_range_set, 2);
        ASSERT_TRUE(program.num_kernels() == 1);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(false, this->devices_.at(id), program, core_to_rt_args));

        std::vector<uint32_t> second_runtime_args = {202, 505};
        SetRuntimeArgs(program, 0, first_core_range, second_runtime_args);
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        for (auto x = first_core_range.start_coord.x; x <= first_core_range.end_coord.x; x++) {
            for (auto y = first_core_range.start_coord.y; y <= first_core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = second_runtime_args;
            }
        }
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(false, this->devices_.at(id), program, core_to_rt_args));

        auto program2 = unit_tests::runtime_args::initialize_program_data_movement_rta(this->devices_.at(id), core_range_set, 4, true);
        // Set common runtime args, automatically sent to all cores used by kernel.
        std::vector<uint32_t> common_runtime_args = {303, 606, 999, 1234};
        SetCommonRuntimeArgs(program2, 0, common_runtime_args);
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program2);
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program2);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(false, this->devices_.at(id), program2, core_to_rt_args, common_runtime_args));
    }
}

TEST_F(DeviceFixture, LegallyModifyRTArgsCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};
        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set, initial_runtime_args.size(), common_runtime_args.size());
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, 0, common_runtime_args);

        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(true, this->devices_.at(id), program, core_to_rt_args, common_runtime_args));
    }
}

// Don't cover all cores of kernel with SetRuntimeArgs. Verify that correct offset used to access common runtime args.
TEST_F(DeviceFixture, SetRuntimeArgsSubsetOfCoresCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};

        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set, initial_runtime_args.size(), common_runtime_args.size());
        SetRuntimeArgs(program, 0, first_core_range, initial_runtime_args); // First core range only.

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto x = first_core_range.start_coord.x; x <= first_core_range.end_coord.x; x++) {
            for (auto y = first_core_range.start_coord.y; y <= first_core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                core_to_rt_args[logical_core] = initial_runtime_args;
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, 0, common_runtime_args);
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(true, this->devices_.at(id), program, core_to_rt_args, common_runtime_args));
    }
}

// Different unique runtime args per core. Not overly special, but verify that it works.
TEST_F(DeviceFixture, SetRuntimeArgsUniqueValuesCompute) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};
        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set, 2, common_runtime_args.size());

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    // Generate an rt arg val based on x and y.
                    uint32_t val_offset = x * 100 + y * 10;
                    std::vector<uint32_t> initial_runtime_args = {101 + val_offset, 202 + val_offset};
                    SetRuntimeArgs(program, 0, logical_core, initial_runtime_args);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, 0, common_runtime_args);

        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(true, this->devices_.at(id), program, core_to_rt_args, common_runtime_args));
    }
}

// Some cores have more unique runtime args than others. Unused in kernel, but API supports it, so verify it works and that
// common runtime args are appropriately offset by amount from core(s) with most unique runtime args.
TEST_F(DeviceFixture, SetRuntimeArgsVaryingLengthPerCore) {

    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};

        // Figure out max number of unique runtime args across all cores, so kernel
        // can attempt to increment that many unique rt args per core. Kernels
        // with fewer will just increment unused memory, no big deal.
        uint32_t max_unique_rt_args = 0;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    uint32_t num_rt_args = 2 + x + y;
                    max_unique_rt_args = std::max(max_unique_rt_args, num_rt_args);
                }
            }
        }

        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set, max_unique_rt_args, common_runtime_args.size());

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    // Generate rt args length and val based on x,y arbitrarily.
                    uint32_t val_offset = x * 100 + y * 10;
                    uint32_t num_rt_args = 2 + x + y;
                    std::vector<uint32_t> initial_runtime_args;
                    for (uint32_t i = 0; i < num_rt_args; i++) {
                        initial_runtime_args.push_back(101 + val_offset + (i * 66));
                    }
                    SetRuntimeArgs(program, 0, logical_core, initial_runtime_args);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }

        // Set common runtime args, automatically sent to all cores used by kernel.
        SetCommonRuntimeArgs(program, 0, common_runtime_args);

        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        EXPECT_TRUE(unit_tests::runtime_args::verify_results(true, this->devices_.at(id), program, core_to_rt_args, common_runtime_args));
    }
}

// Too many unique and common runtime args, overflows allowed space and throws expected exception from both unique/common APIs.
TEST_F(DeviceFixture, IllegalTooManyRuntimeArgs) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        CoreRange first_core_range(CoreCoord(1, 1), CoreCoord(2, 2));
        CoreRangeSet core_range_set(first_core_range);
        auto program = unit_tests::runtime_args::initialize_program_compute(this->devices_.at(id), core_range_set, 0, 0); // Kernel isn't run here.

        // Set 100 unique args, then try to set 300 common args and fail.
        std::vector<uint32_t> initial_runtime_args(100);
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);
        std::vector<uint32_t> common_runtime_args(300);
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program, 0, common_runtime_args));

        // Set 100 common args, then try to set another 300 unique args and fail.
        std::vector<uint32_t> more_common_runtime_args(100);
        SetCommonRuntimeArgs(program, 0, more_common_runtime_args);
        std::vector<uint32_t> more_unique_args(300);
        EXPECT_ANY_THROW(SetRuntimeArgs(program, 0, core_range_set, more_unique_args));
    }
}

TEST_F(DeviceFixture, IllegallyModifyRTArgs) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        // First run the program with the initial runtime args
        CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(5, 5));
        CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});
        auto program = unit_tests::runtime_args::initialize_program_data_movement_rta(this->devices_.at(id), core_range_set, 2);
        ASSERT_TRUE(program.num_kernels() == 1);
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program, 0, core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        detail::WriteRuntimeArgsToDevice(this->devices_.at(id), program);
        tt_metal::detail::LaunchProgram(this->devices_.at(id), program);
        ASSERT_TRUE(unit_tests::runtime_args::verify_results(false, this->devices_.at(id), program, core_to_rt_args));

        std::vector<uint32_t> invalid_runtime_args = {303, 404, 505};
        EXPECT_ANY_THROW(SetRuntimeArgs(program, 0, first_core_range, invalid_runtime_args));

        // Cannot modify number of common runtime args either.
        std::vector<uint32_t> common_runtime_args = {11, 22, 33, 44};
        SetCommonRuntimeArgs(program, 0, common_runtime_args);
        std::vector<uint32_t> illegal_common_runtime_args = {0, 1, 2, 3, 4, 5};
        EXPECT_ANY_THROW(SetCommonRuntimeArgs(program, 0, illegal_common_runtime_args));

    }
}

}  // namespace unit_tests::runtime_args
