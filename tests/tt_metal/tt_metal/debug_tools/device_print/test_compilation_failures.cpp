// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <string>

#include "debug_tools_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

class DevicePrintFailuresFixture : public DevicePrintFixture {
public:
    void TestCompileKernelFailure(
        const std::string& kernel_path,
        const std::string& expected_error_message,
        stl::Span<const uint32_t> runtime_args = {}) {
        try {
            CompileKernel(kernel_path, runtime_args);
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

TEST_F(DevicePrintFailuresFixture, MixedPlaceholders) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/mixed_placeholders.cpp",
        "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");
}

TEST_F(DevicePrintFailuresFixture, InvalidPlaceholderSyntax) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/invalid_placeholder_syntax.cpp",
        "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");
}

TEST_F(DevicePrintFailuresFixture, InvalidPlaceholderIndex) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/invalid_placeholder_index.cpp",
        "Placeholder index exceeds number of arguments");
}

TEST_F(DevicePrintFailuresFixture, NotAllArgumentsReferenced) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/not_all_arguments_referenced.cpp",
        "All arguments must be referenced when using indexed placeholders");
}

TEST_F(DevicePrintFailuresFixture, TooManyArguments) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/too_many_arguments.cpp",
        "Number of {} placeholders must match number of arguments");
}

TEST_F(DevicePrintFailuresFixture, NotEnoughArguments) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/not_enough_arguments.cpp",
        "Number of {} placeholders must match number of arguments");
}
