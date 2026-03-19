// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Parses debug print data from device buffers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "device/device_impl.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include "hostdevcommon/dprint_common.h"
#include "hostdev/device_print_common.h"
#include "hostdev/device_print_structures.h"

namespace tt::tt_metal {

class DPrintParser {
public:
    struct ParseResult {
        std::vector<std::string> completed_lines;
        size_t bytes_consumed{};
    };

    explicit DPrintParser(std::string line_prefix = "");
    ParseResult parse(const uint8_t* data, size_t len);
    std::string flush();

private:
    std::string line_prefix_;
    std::ostringstream intermediate_stream_;
    DPrintTypeID prev_type_{DPrintTypeID_Count};
    char most_recent_setw_{0};

    // Helper methods (from dprint_server.cpp anonymous namespace)
    static float make_float(uint8_t exp_bit_count, uint8_t mantissa_bit_count, uint32_t data);
    static void AssertSize(uint8_t sz, uint8_t expected_sz);
    static bool StreamEndsWithNewlineChar(const std::ostringstream* stream);
    static void ResetStream(std::ostringstream* stream);

    void PrintTileSlice(const uint8_t* ptr);
    void PrintTensixRegisterData(int setwidth, uint32_t datum, uint16_t data_format);
    void PrintTypedUint32Array(
        int setwidth,
        uint32_t raw_element_count,
        const uint32_t* data,
        TypedU32_ARRAY_Format force_array_type = TypedU32_ARRAY_Format_INVALID);

    std::string get_completed_line();
};

class DevicePrintParser {
    using DevicePrintStringInfo = device_print_detail::structures::DevicePrintStringInfo32;

public:
    DevicePrintParser(const DevicePrintParser&) = delete;
    DevicePrintParser& operator=(const DevicePrintParser&) = delete;

    using ArgumentValue =
        std::variant<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>;
    struct FormatMessageBuffer {
        fmt::memory_buffer buffer;
        std::vector<ArgumentValue> argument_values;
    };

    static std::shared_ptr<DevicePrintParser> get_parser_for_elf(const std::string& elf_path);
    std::string_view format_message(
        uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer);

private:
    DevicePrintParser(const std::string& elf_path);

    struct FormatPlaceholderInfo {
        uint32_t arg_id;
        char type_id;
        std::string_view format_spec;  // The part after ':' in the format string, if it exists, including ':' itself.
        std::string fmt_format;        // The full format to be used in fmt::format, e.g. "{0:08x}"
    };

    struct ParsedStringInfo {
        std::string_view format_string;
        std::string_view file;
        uint32_t line = 0;
        std::vector<std::string> plain_text_parts;  // The parts of the format string that are plain text, split by the
                                                    // placeholders. Size is one more than the number of placeholders.
        std::vector<FormatPlaceholderInfo> placeholders;
        std::vector<char> argument_types;
        uint32_t arguments_size = 0;
    };

    ParsedStringInfo* get_string_info(uint32_t info_id);

    static std::size_t get_argument_size_from_type_id(char type_id);
    static ArgumentValue read_argument_from_payload(
        char type_id, std::span<const std::byte> payload_bytes, std::size_t& offset);
    static std::pair<std::vector<std::string>, std::vector<FormatPlaceholderInfo>> parse_format_string(
        std::string_view format_str);
    static std::optional<FormatPlaceholderInfo> parse_placeholder(std::string_view format_str, std::size_t& pos);
    static void read_arguments_from_payload(
        std::span<char> argument_types,
        std::span<const std::byte> payload_bytes,
        std::vector<ArgumentValue>& arguments);
    static std::string_view format_message(
        ParsedStringInfo& string_info, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer);

    std::string elf_path;
    ll_api::ElfFile elf_file;
    std::span<std::byte> format_strings_info_bytes;
    uint64_t format_strings_info_address;
    std::span<std::byte> format_strings_bytes;
    uint64_t format_strings_address;
    DevicePrintStringInfo* string_info_ptr = nullptr;
    size_t string_info_size = 0;
    std::vector<ParsedStringInfo> parsed_string_info;
    static std::map<std::string, std::weak_ptr<DevicePrintParser>> parser_cache;
    friend struct DevicePrintParserDeleter;
};

}  // namespace tt::tt_metal
