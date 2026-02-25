// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace ll_api;
using namespace std::string_view_literals;

struct DevicePrintStringInfo {
    std::uint32_t format_string_ptr;
    std::uint32_t file;
    std::uint32_t line;
};

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
