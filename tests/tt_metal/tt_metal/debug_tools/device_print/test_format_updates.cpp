// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "debug_tools_fixture.hpp"
#include "hostdev/device_print_structures.h"
#include "elf_file.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace std::string_view_literals;

namespace {

using StringInfo32 = device_print_detail::structures::DevicePrintStringInfo32;
using StringInfo64 = device_print_detail::structures::DevicePrintStringInfo64;

template <typename InfoT>
std::vector<StringInfo64> ReadStringInfos(std::span<const std::byte> info_bytes) {
    const auto* entries = reinterpret_cast<const InfoT*>(info_bytes.data());
    const size_t count = info_bytes.size() / sizeof(InfoT);

    std::vector<StringInfo64> infos;
    infos.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        infos.push_back(StringInfo64{
            .format_string_ptr = entries[i].format_string_ptr,
            .file = entries[i].file,
            .line = entries[i].line,
            .padding = entries[i].padding});
    }
    return infos;
}

}  // namespace

class DevicePrintFormatUpdatesFixture : public DevicePrintFixture {
public:
    void TestFormatUpdate(
        const std::string& kernel_path,
        stl::Span<std::string_view> expected_format_messages,
        stl::Span<const uint32_t> runtime_args = {}) {
        const std::string elf_file_path = CompileKernel(kernel_path, runtime_args);

        // Same reader the DEVICE_PRINT server uses (DevicePrintParser::get_parser_for_elf). It also
        // reports the ELF's pointer size, which is what selects the matching string-info layout:
        // the entries are pointer-sized, so they are 4 bytes wide on Wormhole/Blackhole and 8 on
        // Quasar.
        ttexalens::native_elf::ElfFile elf(elf_file_path);

        const auto* info_section = elf.get_section_by_name(".device_print_strings_info");
        const auto* strings_section = elf.get_section_by_name(".device_print_strings");
        ASSERT_NE(info_section, nullptr);
        ASSERT_NE(strings_section, nullptr);

        const std::span<const std::byte> format_strings_info_bytes = info_section->data();
        const std::span<const std::byte> format_strings_bytes = strings_section->data();
        const uint64_t format_strings_address = strings_section->address();
        ASSERT_FALSE(format_strings_info_bytes.empty());
        ASSERT_FALSE(format_strings_bytes.empty());

        const std::vector<StringInfo64> string_infos = elf.get_pointer_size() == 8
                                                           ? ReadStringInfos<StringInfo64>(format_strings_info_bytes)
                                                           : ReadStringInfos<StringInfo32>(format_strings_info_bytes);

        // An entry's format_string_ptr and file are both addresses into the strings section.
        const auto string_at = [&](uint64_t address) {
            return std::string_view(
                reinterpret_cast<const char*>(format_strings_bytes.data() + (address - format_strings_address)));
        };

        for (const auto& expected_format_message : expected_format_messages) {
            const bool found = std::ranges::any_of(string_infos, [&](const StringInfo64& info) {
                return string_at(info.format_string_ptr) == expected_format_message &&
                       string_at(info.file).ends_with(kernel_path);
            });
            if (!found) {
                FAIL() << "Expected format string not found: " << expected_format_message;
            }
        }
    }
};

TEST_F(DevicePrintFormatUpdatesFixture, PrintSimpleString) {
    std::vector<std::string_view> messages = {
        "Hello world!\n"sv,
        "First line.\nSecond line.\n"sv,
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

// Enum arguments are serialized as their base type on device, with the format string updated to contain
// /e_<base_type_char>_<fully_qualified_enum_type_name> as the type specifier.
// The '#' alternate form flag encodes "print full enum type name including value name" on the host side.
TEST_F(DevicePrintFormatUpdatesFixture, PrintEnumValue) {
    std::vector<std::string_view> messages = {
        "Enum1 value: {0,/e_I_test::deep::Enum1}\n"sv,
        "Enum1 full name value: {0,/e_I_test::deep::Enum1:#}\n"sv,
        "Enum1 value: {0,/e_I_test::deep::Enum1}\n"sv,
        "Enum1 full name value: {0,/e_I_test::deep::Enum1:#}\n"sv,
        "Enum1 unrecognized value: {0,/e_I_test::deep::Enum1}\n"sv,
        "Enum1 full name unrecognized value: {0,/e_I_test::deep::Enum1:#}\n"sv,
        "Enum2 value: {0,/e_I_test_shallow::Enum2}\n"sv,
        "Enum2 full name value: {0,/e_I_test_shallow::Enum2:#}\n"sv,
        "EnumClass value: {0,/e_B_EnumClass}\n"sv,
        "EnumClass full name value: {0,/e_B_EnumClass:#}\n"sv,
        "BitEnum value: {0,/E_I_flags::BitEnum}\n"sv,
        "BitEnum full name value: {0,/E_I_flags::BitEnum:#}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_enum_value.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintBuiltinTypes) {
    std::vector<std::string_view> messages = {
        "i={0,i}\n"sv,
        "unknown={0,i}\n"sv,
        "u={0,I}\n"sv,
        "ll={0,q}\n"sv,
        "ull={0,Q}\n"sv,
        "s={0,h}\n"sv,
        "us={0,H}\n"sv,
        "cvllu={0,Q}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_builtin_types.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintStringTypes) {
    std::vector<std::string_view> messages = {
        "Sample string: {0,s}\n"sv,
        "Compile time string: {0,s}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_string_types.cpp", ttsl::make_span(messages));
}

// dp_type_name_t<T> carries no data: the type name is stored in .device_print_strings and the
// argument is the pointer to it, so it uses the same type specifier as CTSTR.
TEST_F(DevicePrintFormatUpdatesFixture, PrintTypeName) {
    std::vector<std::string_view> messages = {
        "builtin type: {0,s}\n"sv,
        "struct type: {0,s}\n"sv,
        "enum type: {0,s}\n"sv,
        "template type: {0,s}\n"sv,
        "decltype: {0,s}\n"sv,
        "with value: {0,s} = {1,f}\n"sv,
    };

    TestFormatUpdate(
        "tests/tt_metal/tt_metal/test_kernels/device_print/print_type_name.cpp", ttsl::make_span(messages));
}

TEST_F(DevicePrintFormatUpdatesFixture, PrintReorder) {
    std::vector<std::string_view> messages = {
        "u16_1: {4,H} u16_2: {5,H} u32_1: {0,I} u32_2: {1,I} u32_3: {2,I} u32_4: {3,I}\n"sv,
    };

    TestFormatUpdate("tests/tt_metal/tt_metal/test_kernels/device_print/print_reorder.cpp", ttsl::make_span(messages));
}
