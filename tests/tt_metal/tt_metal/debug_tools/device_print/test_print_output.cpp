// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;

class DevicePrintOutputFixture : public DevicePrintFixture {
public:
    void TestOutput(
        const std::string& kernel_path,
        const std::vector<std::string>& expected_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        for (auto& mesh_device : this->devices_) {
            RunProgram(mesh_device, kernel_path, runtime_args);
            EXPECT_TRUE(FileContainsAllStrings(dprint_file_name, expected_messages));
        }
    }
};

TEST_F(DevicePrintOutputFixture, PrintSingleUintArg) {
    std::vector<uint32_t> runtime_args = {42};
    std::vector<std::string> messages = {
        "Printing uint32_t from arg: 42",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", messages, runtime_args);
}

TEST_F(DevicePrintOutputFixture, PrintBasicTypes) {
    std::vector<std::string> messages = {
        "int8_t: -8",
        "uint8_t: 8",
        "int16_t: -16",
        "uint16_t: 16",
        "int32_t: -32",
        "uint32_t: 32",
        "int64_t: -64",
        "uint64_t: 64",
        "float: 3.14",
        "double: 6.28",
        "bool: true",
        "bf4_t: 0.5",
        "bf8_t: 0.375",
        "bf16_t: 0.122558594",
        "Reordered args: true -16 -32 -64",
        "Reordered args: true -16 -32 -64",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", messages);
}

TEST_F(DevicePrintOutputFixture, PrintWithFormatSpecified) {
    std::vector<std::string> messages = {
        "int8_t:         -8",
        "uint8_t: 0B1000",
        "int16_t: -16       ",
        "uint16_t: 0X10",
        "int32_t:    -32    ",
        "uint32_t: 0x20",
        "int64_t: -64",
        "uint64_t: 0X000040",
        "float: 3.14",
        "double: 6.28000",
        "bool: true",
    };

    TestOutput("tests/tt_metal/tt_metal/test_kernels/device_print/print_with_format_specified.cpp", messages);
}
