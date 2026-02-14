// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "debug_tools_fixture.hpp"
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

class DevicePrintFormatUpdatesFixture : public DevicePrintFixture {
public:
    void TestFormatUpdate(
        const std::string& kernel_path,
        stl::Span<std::string_view> expected_format_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        const std::string elf_file_path = CompileKernel(kernel_path, runtime_args);

        // Read device_print sections from ELF
        ElfFile elf;
        elf.ReadImage(elf_file_path);

        std::cout << "ELF file read successfully: " << elf_file_path << std::endl;

        const auto& segments = elf.GetSegments();
        ASSERT_FALSE(segments.empty());
        std::vector<std::byte> format_strings_info_bytes;
        uint64_t format_strings_info_address = 0;
        ASSERT_TRUE(elf.GetSectionContents(
            ".device_print_strings_info", format_strings_info_bytes, format_strings_info_address));
        std::vector<std::byte> format_strings_bytes;
        uint64_t format_strings_address = 0;
        ASSERT_TRUE(elf.GetSectionContents(".device_print_strings", format_strings_bytes, format_strings_address));

        // Extract strings from sections
        DevicePrintStringInfo* info_ptr = reinterpret_cast<DevicePrintStringInfo*>(format_strings_info_bytes.data());
        size_t num_messages = format_strings_info_bytes.size() / sizeof(DevicePrintStringInfo);

        for (const auto& expected_format_message : expected_format_messages) {
            bool found = false;
            for (size_t i = 0; i < num_messages; ++i) {
                const DevicePrintStringInfo& info = info_ptr[i];
                const char* format_string = reinterpret_cast<const char*>(
                    format_strings_bytes.data() + (info.format_string_ptr - format_strings_address));
                std::string_view format_str(format_string);
                const char* file_string =
                    reinterpret_cast<const char*>(format_strings_bytes.data() + (info.file - format_strings_address));
                std::string_view file_str(file_string);
                if (format_str == expected_format_message && file_str.ends_with(kernel_path)) {
                    // Found expected format string
                    found = true;
                    break;
                }
            }
            if (!found) {
                FAIL() << "Expected format string not found: " << expected_format_message;
            }
        }
    }
};

TEST_F(DevicePrintFormatUpdatesFixture, PrintSingleUintArg) {
    std::vector<std::string_view> messages = {
        "Printing uint32_t from arg: {0,I}"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintBasicTypes) {
    std::vector<std::string_view> messages = {
        "int8_t: {0,b}"sv,
        "uint8_t: {0,B}"sv,
        "int16_t: {0,h}"sv,
        "uint16_t: {0,H}"sv,
        "int32_t: {0,i}"sv,
        "uint32_t: {0,I}"sv,
        "int64_t: {0,q}"sv,
        "uint64_t: {0,Q}"sv,
        "float: {0,f}"sv,
        "double: {0,d}"sv,
        "bool: {0,?}"sv,
        "Reordered args: {3,?} {2,h} {1,i} {0,q}"sv,
        "Reordered args: {3,?} {2,h} {1,i} {0,q}"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintWithFormatSpecified) {
    std::vector<std::string_view> messages = {
        "int8_t: {0,b: >-10}"sv,
        "uint8_t: {0,B:#B}"sv,
        "int16_t: {0,h: <-10}"sv,
        "uint16_t: {0,H:#X}"sv,
        "int32_t: {0,i: ^-10}"sv,
        "uint32_t: {0,I:#x}"sv,
        "int64_t: {0,q: }"sv,
        "uint64_t: {0,Q:#08X}"sv,
        "float: {0,f:3.3g}"sv,
        "double: {0,d:.5f}"sv,
        "bool: {0,?}"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_with_format_specified.cpp", ttsl::make_span(messages));
}
