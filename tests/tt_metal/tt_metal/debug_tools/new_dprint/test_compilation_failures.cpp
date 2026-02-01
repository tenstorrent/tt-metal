// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <string>

#include "debug_tools_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

class NewDPrintFailuresFixture : public tt::tt_metal::DebugToolsMeshFixture {
public:
    void TestCompileKernelFailure(
        const std::string& kernel_path,
        const std::string& expected_error_message,
        stl::Span<const uint32_t> runtime_args = {}) {
        // Get the first available mesh device
        auto mesh_device = this->devices_.at(0);

        // Set up program
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // This tests prints only on a single core
        constexpr CoreCoord core = {0, 0};  // Print on first core only
        KernelHandle kernel_handle = CreateKernel(
            program_,
            kernel_path,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program_, kernel_handle, core, runtime_args);
        try {
            RunProgram(mesh_device, workload);
        } catch (std::runtime_error& e) {
            std::string error_msg = e.what();
            if (error_msg.find(expected_error_message) != std::string::npos) {
                // Expected error found
                return;
            }
            throw;
        }
        throw std::runtime_error("Expected kernel compilation to fail, but it succeeded.");
    }
};

TEST_F(NewDPrintFailuresFixture, MixedPlaceholders) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/mixed_placeholders.cpp",
        "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");
}

TEST_F(NewDPrintFailuresFixture, InvalidPlaceholderSyntax) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/invalid_placeholder_syntax.cpp",
        "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");
}

TEST_F(NewDPrintFailuresFixture, InvalidPlaceholderIndex) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/invalid_placeholder_index.cpp",
        "Placeholder index exceeds number of arguments");
}

TEST_F(NewDPrintFailuresFixture, NotAllArgumentsReferenced) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/not_all_arguments_referenced.cpp",
        "All arguments must be referenced when using indexed placeholders");
}

TEST_F(NewDPrintFailuresFixture, TooManyArguments) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/too_many_arguments.cpp",
        "Number of {} placeholders must match number of arguments");
}

TEST_F(NewDPrintFailuresFixture, NotEnoughArguments) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/failures/not_enough_arguments.cpp",
        "Number of {} placeholders must match number of arguments");
}
