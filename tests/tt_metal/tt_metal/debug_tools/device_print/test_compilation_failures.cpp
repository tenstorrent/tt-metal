// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>
#include <string>

#include "debug_tools_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

class DevicePrintFailuresFixture : public DevicePrintFixture {
public:
    void TestCompileKernelFailure(
        const std::string& kernel_path,
        const std::string& expected_error_message,
        ttsl::Span<const uint32_t> runtime_args = {}) {
        try {
            CompileKernel(kernel_path, runtime_args);
        } catch (std::runtime_error& e) {
            // The build log also echoes the source of the print macro, every check's message included, so the
            // message counts only on a line that reports an error.
            std::istringstream build_log(e.what());
            for (std::string line; std::getline(build_log, line);) {
                if (line.find("error:") != std::string::npos &&
                    line.find(expected_error_message) != std::string::npos) {
                    return;
                }
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

TEST_F(DevicePrintFailuresFixture, OldStyleDprint) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/old_style_dprint.cpp",
        "Old style DPRINT is deprecated. Use DPRINT(format, ...) instead.");
}

// Known issue: a spec isn't checked against its argument when the kernel compiles, so today this compiles. Disabled
// until that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_SpecNotSuitingArgument) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/spec_not_suiting_argument.cpp",
        "invalid format specifier");
}

// Known issue: a spec isn't checked against its argument when the kernel compiles, and a char is sent as uint8_t, so
// today this compiles and prints "097". Disabled until that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_SpecNotSuitingChar) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/spec_not_suiting_char.cpp",
        "invalid format specifier");
}

TEST_F(DevicePrintFailuresFixture, PrintTooBigForBuffer) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/print_too_big_for_buffer.cpp",
        "does not fit in the print buffer");
}

// Known issue: a spec isn't checked against a tile when the kernel compiles, so today this compiles. Disabled until
// that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_SpecSuitingNoNumberOnTile) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/spec_suiting_no_number_on_tile.cpp",
        "invalid format specifier");
}

// Known issue: a spec isn't checked against a tile when the kernel compiles, so today this compiles. Disabled until
// that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_SpecSuitingNoNumberCombinedOnTile) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/spec_suiting_no_number_combined_on_tile.cpp",
        "invalid format specifier");
}

// Known issue: a spec isn't checked against a typed array when the kernel compiles, so today this compiles. Disabled
// until that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_SpecSuitingNoNumberOnTypedArray) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/spec_suiting_no_number_on_typed_array.cpp",
        "invalid format specifier");
}

// Known issue: today a format that isn't a string literal fails with an unrelated message ("'format' is not
// captured"). Disabled until that is fixed.
TEST_F(DevicePrintFailuresFixture, DISABLED_FormatNotStringLiteral) {
    TestCompileKernelFailure(
        "tests/tt_metal/tt_metal/test_kernels/device_print/failures/format_not_string_literal.cpp",
        "format must be a string literal");
}
