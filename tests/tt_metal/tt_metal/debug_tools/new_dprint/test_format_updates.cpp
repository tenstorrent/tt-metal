// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <string>
#include <vector>

#include "debug_tools_fixture.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace ll_api;

struct DPrintStringInfo {
    std::uint32_t format_string_ptr;
    std::uint32_t file;
    std::uint32_t line;
};

class NewDPrintFormatUpdatesFixture : public NewDPrintFixture {
public:
    void TestFormatUpdate(
        const std::string& kernel_path,
        const std::string& expected_format_message,
        stl::Span<const uint32_t> runtime_args = {}) {
        const std::string elf_file_path = CompileKernel(kernel_path, runtime_args);

        // Read dprint sections from ELF
        ElfFile elf;
        elf.ReadImage(elf_file_path);

        const auto& segments = elf.GetSegments();
        ASSERT_FALSE(segments.empty());
        std::vector<std::byte> format_strings_info_bytes;
        uint64_t format_strings_info_address = 0;
        ASSERT_TRUE(
            elf.GetSectionContents(".dprint_strings_info", format_strings_info_bytes, format_strings_info_address));
        std::vector<std::byte> format_strings_bytes;
        uint64_t format_strings_address = 0;
        ASSERT_TRUE(elf.GetSectionContents(".dprint_strings", format_strings_bytes, format_strings_address));

        // Extract strings from sections
        DPrintStringInfo* info_ptr = reinterpret_cast<DPrintStringInfo*>(format_strings_info_bytes.data());
        size_t num_messages = format_strings_info_bytes.size() / sizeof(DPrintStringInfo);

        for (size_t i = 0; i < num_messages; ++i) {
            const DPrintStringInfo& info = info_ptr[i];
            const char* format_string = reinterpret_cast<const char*>(
                format_strings_bytes.data() + (info.format_string_ptr - format_strings_address));
            std::string format_str(format_string);
            const char* file_string =
                reinterpret_cast<const char*>(format_strings_bytes.data() + (info.file - format_strings_address));
            std::string file_str(file_string);
            if (format_str == expected_format_message && file_str.ends_with(kernel_path)) {
                // Found expected format string
                return;
            }
        }
        FAIL() << "Expected format string not found: " << expected_format_message;
    }
};

TEST_F(NewDPrintFormatUpdatesFixture, PrintOneInt) {
    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/new_dprint/print_one_int.cpp", "Printing int from arg: {0:I}");
}
