#include <algorithm>
#include <functional>
#include <random>

#include "doctest.h"
#include "single_device_fixture.hpp"
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
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default
    );

    CompileProgram(device, program);
    return std::move(program);
}

bool verify_result(Device *device, const Program &program, const std::map<CoreCoord, std::vector<uint32_t>> &core_to_rt_args) {
    bool pass = true;
    auto get_runtime_arg_addr = [](Kernel *kernel) {
        uint32_t result_base = 0;
        switch (kernel->kernel_type()) {
            case KernelType::DataMovement: {
                auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel);
                switch (dm_kernel->data_movement_processor()) {
                    case DataMovementProcessor::RISCV_0: {
                        result_base = BRISC_L1_ARG_BASE;
                    }
                    break;
                    case DataMovementProcessor::RISCV_1: {
                        result_base = NCRISC_L1_ARG_BASE;
                    }
                    break;
                    default:
                        log_assert(false, "Unsupported data movement processor");
                    }
            }
            break;
            default:
                log_assert(false, "Only BRISC and NCRISC have runtime arg support");
        }
        return result_base;
    };

    CHECK(program.kernels().size() == 3); //2 Blanks get auto-populated even though we added 1 kernel into program
    auto processor = program.kernels().at(0)->processor();
    auto rt_arg_addr = get_runtime_arg_addr(program.kernels().at(0));

    for (const auto &kernel : program.kernels()) {
        auto processor = kernel->processor();
        for (const auto &[logical_core, rt_args] : kernel->runtime_args()) {
            auto expected_rt_args = core_to_rt_args.at(logical_core);
            CHECK(rt_args == expected_rt_args);
            std::vector<uint32_t> written_args;
            tt_metal::ReadFromDeviceL1(device, logical_core, rt_arg_addr, rt_args.size()*sizeof(uint32_t), written_args);
            bool got_expected_result = rt_args == written_args;
            CHECK(got_expected_result);
            pass &= got_expected_result;
        }
    }
    return pass;
}

} // namespace unit_tests::runtime_args

TEST_SUITE(
    "Initialize&ModifyRuntimeArgs" *
    doctest::description("Tests that set runtime args and change them before running the same program")) {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Legally and Illegally Modify RT Args") {
        // First run the program with the initial runtime args
        CoreRange first_core_range = {.start = CoreCoord(0, 0), .end = CoreCoord(1, 1)};
        CoreRange second_core_range = {.start = CoreCoord(3, 3), .end = CoreCoord(5, 5)};
        CoreRangeSet core_range_set({first_core_range, second_core_range});
        auto program = unit_tests::runtime_args::init_compile_and_configure_program(this->device_, core_range_set);
        REQUIRE(program.kernels().size() == 3); //2 Blanks get auto-populated even though we added 1 kernel into program
        std::vector<uint32_t> initial_runtime_args = {101, 202};
        SetRuntimeArgs(program.kernels().at(0), core_range_set, initial_runtime_args);

        std::map<CoreCoord, std::vector<uint32_t>> core_to_rt_args;
        for (auto core_range : core_range_set.ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = initial_runtime_args;
                }
            }
        }
        WriteRuntimeArgsToDevice(this->device_, program);
        REQUIRE(unit_tests::runtime_args::verify_result(this->device_, program, core_to_rt_args));

        SUBCASE("Legal modification of RT args") {
            std::vector<uint32_t> second_runtime_args = {303, 606};
            SetRuntimeArgs(program.kernels().at(0), first_core_range, second_runtime_args);
            WriteRuntimeArgsToDevice(this->device_, program);
            for (auto x = first_core_range.start.x; x <= first_core_range.end.x; x++) {
                for (auto y = first_core_range.start.y; y <= first_core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    core_to_rt_args[logical_core] = second_runtime_args;
                }
            }
            CHECK(unit_tests::runtime_args::verify_result(this->device_, program, core_to_rt_args));
        }

        SUBCASE("Illegal modification of RT args") {
            std::vector<uint32_t> invalid_runtime_args = {303, 404, 505};
            CHECK_THROWS_WITH(
                SetRuntimeArgs(program.kernels().at(0), first_core_range, invalid_runtime_args),
                doctest::Contains("Illegal Runtime Args")
            );
        }
    }
}
