// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include "hostdev/device_print_structures.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace ll_api;
using namespace std::string_view_literals;

using DevicePrintStringInfo = device_print_detail::structures::DevicePrintStringInfo32;

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
        uint64_t format_strings_info_address = 0;
        std::span<std::byte> format_strings_info_bytes =
            elf.GetSectionContents(".device_print_strings_info", format_strings_info_address);
        ASSERT_FALSE(format_strings_info_bytes.empty());
        uint64_t format_strings_address = 0;
        std::span<std::byte> format_strings_bytes =
            elf.GetSectionContents(".device_print_strings", format_strings_address);
        ASSERT_FALSE(format_strings_bytes.empty());

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

TEST_F(DevicePrintFormatUpdatesFixture, PrintSimpleString) {
    std::vector<std::string_view> messages = {
        "Hello world!\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_simple_string.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintSingleUintArg) {
    std::vector<std::string_view> messages = {
        "Printing uint32_t from arg: {0,I}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_single_uint_arg.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintFactorial) {
    std::vector<std::string_view> messages = {
        "factorial({0,I}) = {1,I}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_factorial.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintBasicTypes) {
    std::vector<std::string_view> messages = {
        "int8_t: {0,b}\n"sv,
        "uint8_t: {0,B}\n"sv,
        "int16_t: {0,h}\n"sv,
        "uint16_t: {0,H}\n"sv,
        "int32_t: {0,i}\n"sv,
        "uint32_t: {0,I}\n"sv,
        "int64_t: {0,q}\n"sv,
        "uint64_t: {0,Q}\n"sv,
        "float: {0,f}\n"sv,
        "double: {0,d}\n"sv,
        "bool: {0,?}\n"sv,
        "bf4_t: {0,e}\n"sv,
        "bf8_t: {0,E}\n"sv,
        "bf16_t: {0,w}\n"sv,
        "Reordered args: {3,?} {2,h} {1,i} {0,q}\n"sv,
        "Reordered args: {3,?} {2,h} {1,i} {0,q}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_basic_types.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintWithFormatSpecified) {
    std::vector<std::string_view> messages = {
        "int8_t: {0,b: >-10}\n"sv,
        "uint8_t: {0,B:#B}\n"sv,
        "int16_t: {0,h: <-10}\n"sv,
        "uint16_t: {0,H:#X}\n"sv,
        "int32_t: {0,i: ^-10}\n"sv,
        "uint32_t: {0,I:#x}\n"sv,
        "int64_t: {0,q: }\n"sv,
        "uint64_t: {0,Q:#08X}\n"sv,
        "float: {0,f:3.3g}\n"sv,
        "double: {0,d:.5f}\n"sv,
        "bool: {0,?}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_with_format_specified.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintAllArgumentSizes) {
    std::vector<std::string_view> messages = {
        "No arguments\n"sv,
        "1 argument: {0,I}\n"sv,
        "2 arguments: {0,I} {1,I}\n"sv,
        "3 arguments: {0,I} {1,I} {2,I}\n"sv,
        "4 arguments: {0,I} {1,I} {2,I} {3,I}\n"sv,
        "5 arguments: {0,I} {1,I} {2,I} {3,I} {4,I}\n"sv,
        "6 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I}\n"sv,
        "7 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I}\n"sv,
        "8 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I}\n"sv,
        "9 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I}\n"sv,
        "10 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I}\n"sv,
        "11 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I}\n"sv,
        "12 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I}\n"sv,
        "13 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I}\n"sv,
        "14 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I}\n"sv,
        "15 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I}\n"sv,
        "16 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I}\n"sv,
        "17 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I}\n"sv,
        "18 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I}\n"sv,
        "19 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I}\n"sv,
        "20 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I}\n"sv,
        "21 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I}\n"sv,
        "22 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I}\n"sv,
        "23 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I}\n"sv,
        "24 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I}\n"sv,
        "25 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I}\n"sv,
        "26 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I}\n"sv,
        "27 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I}\n"sv,
        "28 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I} {27,I}\n"sv,
        "29 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I} {27,I} {28,I}\n"sv,
        "30 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I} {27,I} {28,I} {29,I}\n"sv,
        "31 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I} {27,I} {28,I} {29,I} {30,I}\n"sv,
        "32 arguments: {0,I} {1,I} {2,I} {3,I} {4,I} {5,I} {6,I} {7,I} {8,I} {9,I} {10,I} {11,I} {12,I} {13,I} {14,I} {15,I} {16,I} {17,I} {18,I} {19,I} {20,I} {21,I} {22,I} {23,I} {24,I} {25,I} {26,I} {27,I} {28,I} {29,I} {30,I} {31,I}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_all_argument_sizes.cpp", ttsl::make_span(messages));
}
