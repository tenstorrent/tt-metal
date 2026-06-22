// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <enchantum/scoped.hpp>
#include "device/device_impl.hpp"
#include "impl/data_format/blockfloat_common.hpp"
#include <tt_stl/assert.hpp>

#include "fmt/base.h"
#include "hostdevcommon/dprint_common.h"
#include "hostdev/device_print_structures.h"
#include "tt_backend_api_types.hpp"

#include "elf_file.hpp"
#include "dwarf_die.hpp"
#include "callstack.hpp"

using std::string;
using namespace std::literals;

namespace tt::tt_metal {

inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

// Create a float from a given bit pattern, given the number of bits for the exponent and mantissa.
// Assumes the following order of bits in the input data:
//   [sign bit][mantissa bits][exponent bits]
inline float make_float(uint8_t exp_bit_count, uint8_t mantissa_bit_count, uint32_t data) {
    int sign = (data >> (exp_bit_count + mantissa_bit_count)) & 0x1;
    const int exp_mask = (1 << (exp_bit_count)) - 1;
    int exp_bias = (1 << (exp_bit_count - 1)) - 1;
    int exp_val = (data & exp_mask) - exp_bias;
    const int mantissa_mask = ((1 << mantissa_bit_count) - 1) << exp_bit_count;
    int mantissa_val = (data & mantissa_mask) >> exp_bit_count;
    // Zero exponent and zero mantissa means zero (IEEE 754 convention)
    if ((data & exp_mask) == 0 && mantissa_val == 0) {
        return sign ? -0.0f : 0.0f;
    }
    float result = 1.0 + ((float)mantissa_val / (float)(1 << mantissa_bit_count));
    result = result * pow(2, exp_val);
    if (sign) {
        result = -result;
    }
    return result;
}

std::map<std::string, std::weak_ptr<DevicePrintParser>> DevicePrintParser::parser_cache;

template <uint8_t PointerSize>
class DevicePrintParserImpl : public DevicePrintParser {
public:
    using pointer_t = std::conditional_t<PointerSize == 4, uint32_t, uint64_t>;
    using DevicePrintStringInfo = std::conditional_t<
        PointerSize == 4,
        device_print_detail::structures::DevicePrintStringInfo32,
        device_print_detail::structures::DevicePrintStringInfo64>;

    explicit DevicePrintParserImpl(const std::string& elf_path, ttexalens::native_elf::ElfFile elf);

    std::string_view format_message(
        uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) override;

    const std::string& get_elf_path() const override { return elf_path; }

private:
    struct EnumInfo {
        std::string type_name;
        std::vector<std::pair<int64_t, std::string>> enumerators;
    };

    struct FormatPlaceholderInfo {
        uint32_t arg_id{};
        char type_id{};
        std::string_view format_spec;
        std::string fmt_format;
        // Enum-specific fields (only used when type_id == '/')
        std::string_view enum_type_name;
        char enum_base_type_id{};
        bool enum_is_flag{};
        bool enum_use_full_name{};
        const EnumInfo* enum_info{};
    };

    struct ParsedStringInfo {
        std::string format_string;
        std::string file;
        uint32_t line{};
        std::vector<std::string> plain_text_parts;
        std::vector<FormatPlaceholderInfo> placeholders;
        std::vector<char> argument_types;
        std::size_t arguments_size{};
    };

    ParsedStringInfo* get_string_info(uint32_t info_id);

    std::size_t get_argument_size_from_type_id(char type_id) const;

    void read_arguments_from_payload(
        std::span<char> argument_types,
        std::span<const std::byte> payload_bytes,
        std::vector<ArgumentValue>& arguments) const;

    ArgumentValue read_argument_from_payload(
        char type_id, std::span<const std::byte> payload_bytes, std::size_t& offset) const;

    static std::pair<std::vector<std::string>, std::vector<FormatPlaceholderInfo>> parse_format_string(
        std::string_view format_str);

    static std::optional<FormatPlaceholderInfo> parse_placeholder(std::string_view format_str, std::size_t& pos);

    std::string_view format_message(
        ParsedStringInfo& string_info, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer);

    const EnumInfo* get_enum_info(std::string_view type_name);

    auto resolve_top_callstack(const TopCallstackInfo& info);
    void format_top_callstack(fmt::memory_buffer& out, const TopCallstackInfo& info);

    std::string elf_path;
    ttexalens::native_elf::ElfFile elf_file;
    std::span<const std::byte> format_strings_info_bytes;
    uint64_t format_strings_info_address{};
    std::span<const std::byte> format_strings_bytes;
    uint64_t format_strings_address{};
    const DevicePrintStringInfo* string_info_ptr{};
    size_t string_info_size{};
    std::vector<ParsedStringInfo> parsed_string_info;
    std::map<std::string, EnumInfo, std::less<>> enum_info_cache_;
};

template <uint8_t PointerSize>
DevicePrintParserImpl<PointerSize>::DevicePrintParserImpl(
    const std::string& elf_path, ttexalens::native_elf::ElfFile elf) :
    elf_path(elf_path), elf_file(std::move(elf)) {
    try {
        const auto* info_section = elf_file.get_section_by_name(".device_print_strings_info");
        const auto* strings_section = elf_file.get_section_by_name(".device_print_strings");
        if (info_section == nullptr || strings_section == nullptr) {
            throw std::runtime_error("ELF is missing the DEVICE_PRINT string sections");
        }
        format_strings_info_bytes = info_section->data();
        format_strings_info_address = info_section->address();
        format_strings_bytes = strings_section->data();
        format_strings_address = strings_section->address();
        string_info_ptr = reinterpret_cast<const DevicePrintStringInfo*>(format_strings_info_bytes.data());
        string_info_size = format_strings_info_bytes.size() / sizeof(DevicePrintStringInfo);
        parsed_string_info.resize(string_info_size);
    } catch (...) {
        // ELF loaded but its DEVICE_PRINT sections are missing/unreadable — degrade to a no-op parser.
        log_warning(tt::LogMetal, "Failed to parse DEVICE_PRINT info from ELF file {}", elf_path);
    }
}

struct DevicePrintParserDeleter {
    void operator()(DevicePrintParser* parser) const {
        DevicePrintParser::parser_cache.erase(parser->get_elf_path());
        delete parser;
    }
};

std::shared_ptr<DevicePrintParser> DevicePrintParser::get_parser_for_elf(const std::string& elf_path) {
    auto cached_parser_it = parser_cache.find(elf_path);
    if (cached_parser_it != parser_cache.end()) {
        if (auto cached_parser = cached_parser_it->second.lock()) {
            return cached_parser;
        }
    }
    // Parse the ELF once here; ttexalens_elf reports the pointer size, which selects the 32-/64-bit
    // parser, and the parsed ElfFile is then handed off to it so the file isn't opened a second time.
    ttexalens::native_elf::ElfFile elf_file(elf_path);
    std::shared_ptr<DevicePrintParser> new_parser;
    if (elf_file.get_pointer_size() == 8) {
        new_parser = std::shared_ptr<DevicePrintParser>(
            new DevicePrintParserImpl<8>(elf_path, std::move(elf_file)), DevicePrintParserDeleter());
    } else {
        new_parser = std::shared_ptr<DevicePrintParser>(
            new DevicePrintParserImpl<4>(elf_path, std::move(elf_file)), DevicePrintParserDeleter());
    }
    parser_cache[elf_path] = new_parser;
    return new_parser;
}

template <uint8_t PointerSize>
std::string_view DevicePrintParserImpl<PointerSize>::format_message(
    uint32_t info_id, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) {
    auto* string_info = get_string_info(info_id);
    if (string_info == nullptr) {
        return {};
    }
    return format_message(*string_info, payload_bytes, buffer);
}

template <uint8_t PointerSize>
typename DevicePrintParserImpl<PointerSize>::ParsedStringInfo* DevicePrintParserImpl<PointerSize>::get_string_info(
    uint32_t info_id) {
    if (info_id >= string_info_size) {
        return nullptr;
    }
    auto& parsed_info = parsed_string_info[info_id];
    if (parsed_info.format_string.empty()) {
        // This entry has not been parsed yet, so parse it now and cache the result.
        uint64_t info_format_ptr = string_info_ptr[info_id].format_string_ptr;
        uint64_t info_file = string_info_ptr[info_id].file;
        uint64_t info_line = string_info_ptr[info_id].line;
        if (info_format_ptr >= format_strings_address &&
            info_format_ptr < format_strings_address + format_strings_bytes.size()) {
            const char* format_string =
                reinterpret_cast<const char*>(format_strings_bytes.data() + (info_format_ptr - format_strings_address));
            parsed_info.format_string = format_string;
            std::tie(parsed_info.plain_text_parts, parsed_info.placeholders) = parse_format_string(format_string);
            if (!parsed_info.placeholders.empty()) {
                uint32_t max_arg_id = 0;
                for (const auto& placeholder : parsed_info.placeholders) {
                    max_arg_id = std::max(max_arg_id, placeholder.arg_id);
                }
                parsed_info.argument_types.resize(max_arg_id + 1);
                for (auto& placeholder : parsed_info.placeholders) {
                    // For long types (type_id == '/'), the serialization uses the base type
                    char serialization_type = placeholder.type_id;
                    if (placeholder.type_id == '/' && placeholder.enum_base_type_id != 0) {
                        // '/e' enum type: serialize as the underlying integer type
                        serialization_type = placeholder.enum_base_type_id;
                        placeholder.enum_info = get_enum_info(placeholder.enum_type_name);
                    }
                    parsed_info.argument_types[placeholder.arg_id] = serialization_type;
                    parsed_info.arguments_size += get_argument_size_from_type_id(serialization_type);
                }
            }
        }
        if (info_file >= format_strings_address && info_file < format_strings_address + format_strings_bytes.size()) {
            const char* file =
                reinterpret_cast<const char*>(format_strings_bytes.data() + (info_file - format_strings_address));
            parsed_info.file = file;
        }
        parsed_info.line = static_cast<uint32_t>(info_line);
    }
    return &parsed_info;
}

template <typename T>
T read_value_from_payload(std::span<const std::byte> payload_bytes, std::size_t& offset) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset + sizeof(T) > payload_bytes.size()) {
        TT_THROW("Payload does not contain enough bytes to read type");
    }
    T value;
    std::memcpy(&value, payload_bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

template <uint8_t PointerSize>
std::size_t DevicePrintParserImpl<PointerSize>::get_argument_size_from_type_id(char type_id) const {
    static std::byte empty_bytes[32];
    std::size_t offset = 0;
    read_argument_from_payload(type_id, std::span<const std::byte>(empty_bytes), offset);
    return offset;
}

template <uint8_t PointerSize>
void DevicePrintParserImpl<PointerSize>::read_arguments_from_payload(
    std::span<char> argument_types,
    std::span<const std::byte> payload_bytes,
    std::vector<ArgumentValue>& arguments) const {
    std::size_t payload_offset = 0;

    arguments.clear();
    arguments.reserve(argument_types.size());
    for (char argument_type : argument_types) {
        arguments.push_back(read_argument_from_payload(argument_type, payload_bytes, payload_offset));
    }
}

template <uint8_t PointerSize>
std::pair<std::vector<std::string>, std::vector<typename DevicePrintParserImpl<PointerSize>::FormatPlaceholderInfo>>
DevicePrintParserImpl<PointerSize>::parse_format_string(std::string_view format_str) {
    std::vector<std::string> plain_text_parts;
    std::vector<FormatPlaceholderInfo> placeholders;
    fmt::memory_buffer current_text;
    for (size_t i = 0; i < format_str.size(); i++) {
        if (format_str[i] == '{' && i + 1 < format_str.size() && format_str[i + 1] == '{') {
            // Escaped '{', add a single '{' to the result and skip the next character.
            current_text.push_back('{');
            i++;
            continue;
        }
        if (format_str[i] == '}' && i + 1 < format_str.size() && format_str[i + 1] == '}') {
            // Escaped '}', add a single '}' to the result and skip the next character.
            current_text.push_back('}');
            i++;
            continue;
        }
        if (format_str[i] == '{') {
            auto placeholder = parse_placeholder(format_str, i);
            if (!placeholder) {
                TT_THROW("Invalid format string: failed to parse placeholder at position {}", i);
            }
            placeholders.push_back(*placeholder);
            plain_text_parts.push_back(std::string(current_text.data(), current_text.size()));
            current_text.clear();
            i--;  // Step back so that the main loop can correctly identify the end of the placeholder
        } else {
            // Regular character, add it to the result.
            current_text.push_back(format_str[i]);
            continue;
        }
    }
    plain_text_parts.push_back(std::string(current_text.data(), current_text.size()));
    return {std::move(plain_text_parts), std::move(placeholders)};
}

template <uint8_t PointerSize>
std::optional<typename DevicePrintParserImpl<PointerSize>::FormatPlaceholderInfo>
DevicePrintParserImpl<PointerSize>::parse_placeholder(std::string_view format_str, std::size_t& pos) {
    if (pos >= format_str.size() || format_str[pos] != '{') {
        return std::nullopt;
    }

    // Start of a placeholder. Read until the closing '}' to extract the placeholder content.
    pos++;  // Skip '{'

    // We are trying to mimic fmtlib format specifiers here, but device already changed it a bit:
    // replacement_field ::= "{" arg_id "," type_id [":" format_spec] "}"
    // arg_id            ::= integer
    // integer           ::= digit+
    // digit             ::= "0"..."9"
    // type_id           ::= short_type | long_type
    // short_type        ::= "a"..."z" | "A"..."Z"         (single character, e.g. 'I' for uint32_t)
    // long_type         ::= "/" sub_type "_" extra_info   ('/' signals a multi-character type descriptor)
    // sub_type          ::= "e"                           (regular enum)
    //                     | "E"                           (flag/bitmask enum with operator|)
    // extra_info        ::= short_type "_" enum_name      (e.g. "I_test::deep::Enum1")
    // But we don't support using identifiers to reduce kernel size, only integers for arg_id.

    // Regarding format_spec:
    // format_spec ::= [[fill]align][sign]["#"]["0"][width]["." precision]["L"][type]
    // fill        ::= <a character other than '{' or '}'>
    // align       ::= "<" | ">" | "^"
    // sign        ::= "+" | "-" | " "
    // width       ::= integer | "{" [arg_id] "}"
    // precision   ::= integer | "{" [arg_id] "}"
    // type        ::= "a" | "A" | "b" | "B" | "c" | "d" | "e" | "E" | "f" | "F" |
    //                 "g" | "G" | "o" | "p" | "s" | "x" | "X" | "?"
    // We don't support using arg_id for width/precision.

    // As everything is verified during kernel compile time, we can parse format_spec just by reading until the closing
    // '}' without needing to fully understand it on the host side.
    uint32_t arg_id = 0;

    // arg_id parsing
    if (!std::isdigit(format_str[pos])) {
        return std::nullopt;
    }
    while (pos < format_str.size() && std::isdigit(format_str[pos])) {
        arg_id = arg_id * 10 + (format_str[pos] - '0');
        pos++;
    }

    // Read type_id (the character after arg_id and ',')
    if (pos >= format_str.size() || format_str[pos] != ',') {
        return std::nullopt;
    }
    pos++;  // Skip ','
    char type_id = format_str[pos++];

    // Check for long type marker: '/' followed by sub-type character.
    // Supported: /e_<base_type>_<enum_name> (regular enum), /E_<base_type>_<enum_name> (flag enum)
    if (type_id == '/' && pos + 3 < format_str.size() && (format_str[pos] == 'e' || format_str[pos] == 'E') &&
        format_str[pos + 1] == '_') {
        bool is_flag_enum = (format_str[pos] == 'E');
        pos += 2;  // Skip 'e_' or 'E_'
        char base_type = format_str[pos++];
        if (pos < format_str.size() && format_str[pos] == '_') {
            pos++;  // Skip second '_'
        }
        // Read enum type name until format spec separator or '}'.
        // A single ':' not followed by ':' marks the start of a format spec.
        // '::' is a C++ namespace separator and is part of the enum type name.
        std::size_t name_start = pos;
        while (pos < format_str.size() && format_str[pos] != '}') {
            if (format_str[pos] == ':' && (pos + 1 >= format_str.size() || format_str[pos + 1] != ':')) {
                break;  // Single ':' = format spec separator
            }
            if (format_str[pos] == ':' && pos + 1 < format_str.size() && format_str[pos + 1] == ':') {
                pos += 2;  // Skip '::' as part of the name
                continue;
            }
            pos++;
        }
        std::string_view enum_type_name = format_str.substr(name_start, pos - name_start);

        // Parse optional format spec (may contain '#' for full name)
        bool use_full_name = false;
        std::size_t format_spec_start = pos;
        while (pos < format_str.size() && format_str[pos] != '}') {
            if (format_str[pos] == '#') {
                use_full_name = true;
            }
            pos++;
        }
        auto format_spec = format_str.substr(format_spec_start, pos - format_spec_start);
        pos++;  // Skip '}'

        // Build fmt_format for the enum string, stripping '#' which is consumed as enum_use_full_name.
        std::string fmt_format = "{0";
        for (char c : format_spec) {
            if (c != '#') {
                fmt_format += c;
            }
        }
        fmt_format += '}';

        FormatPlaceholderInfo info;
        info.arg_id = arg_id;
        info.type_id = '/';  // Mark as long type (enum)
        info.format_spec = format_spec;
        info.fmt_format = std::move(fmt_format);
        info.enum_type_name = enum_type_name;
        info.enum_base_type_id = base_type;
        info.enum_is_flag = is_flag_enum;
        info.enum_use_full_name = use_full_name;
        return info;
    }

    uint32_t format_spec_start = pos;
    while (pos < format_str.size() && format_str[pos] != '}') {
        pos++;
    }
    pos++;  // Skip '}'
    auto format_spec = format_str.substr(format_spec_start, pos - format_spec_start - 1);

    FormatPlaceholderInfo info;
    info.arg_id = arg_id;
    info.type_id = type_id;
    info.format_spec = format_spec;
    info.fmt_format = "{0" + std::string(format_spec) + "}";
    return info;
}

template <uint8_t PointerSize>
std::string_view DevicePrintParserImpl<PointerSize>::format_message(
    ParsedStringInfo& string_info, std::span<const std::byte> payload_bytes, FormatMessageBuffer& buffer) {
    // Iterate over format_str and replace {} with format of payload values.
    buffer.buffer.clear();
    auto format_str = string_info.format_string;
    if (string_info.arguments_size > payload_bytes.size()) {
        log_warning(
            tt::LogMetal,
            "Payload size {} is smaller than expected arguments size {} for format string '{}'",
            payload_bytes.size(),
            string_info.arguments_size,
            format_str);
        return {};
    }
    read_arguments_from_payload(string_info.argument_types, payload_bytes, buffer.argument_values);

    for (size_t i = 0; i < string_info.placeholders.size(); i++) {
        // Append prefix plain text part before the placeholder
        auto& plain_text_part = string_info.plain_text_parts[i];
        buffer.buffer.append(plain_text_part.data(), plain_text_part.data() + plain_text_part.size());

        // Append the formatted argument for the placeholder
        auto& placeholder = string_info.placeholders[i];

        // Do the actual formatting of the argument
        auto format = fmt::runtime(placeholder.fmt_format);

        switch (placeholder.type_id) {
            case 'b':  // int8_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int8_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'B':  // uint8_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint8_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'h':  // int16_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int16_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'H':  // uint16_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint16_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'i':  // int32_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int32_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'I':  // uint32_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint32_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'q':  // int64_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<int64_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'Q':  // uint64_t
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<uint64_t>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'f':  // float
            case 'e':  // bf4_t, but stored as float
            case 'E':  // bf8_t, but stored as float
            case 'w':  // bf16_t, but stored as float
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<float>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 'd':  // double
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<double>(buffer.argument_values[placeholder.arg_id]));
                break;
            case '?':  // bool
                fmt::format_to(
                    std::back_inserter(buffer.buffer),
                    format,
                    std::get<bool>(buffer.argument_values[placeholder.arg_id]));
                break;
            case 't':  // TileSliceDynamic
            {
                const auto& tile_slice = std::get<TileSliceDynamic>(buffer.argument_values[placeholder.arg_id]);

                // Read any error codes and handle accordingly
                tt::CBIndex cb = static_cast<tt::CBIndex>(tile_slice.header.cb_id);
                switch (tile_slice.header.return_code) {
                    case DPrintOK: break;  // Continue to print the tile slice
                    case DPrintErrorBadPointer: {
                        uint32_t cb_ptr_val = tile_slice.header.cb_ptr;
                        uint8_t count = tile_slice.header.data_count;
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Tried printing {}: BAD TILE POINTER (ptr={}, count={})\n",
                            enchantum::scoped::to_string(cb),
                            cb_ptr_val,
                            count);
                        continue;
                    }
                    case DPrintErrorUnsupportedFormat: {
                        tt::DataFormat data_format = static_cast<tt::DataFormat>(tile_slice.header.data_format);
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Tried printing {}: Unsupported data format ({})\n",
                            enchantum::scoped::to_string(cb),
                            data_format);
                        continue;
                    }
                    case DPrintErrorMath:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: MATH core does not support TileSlice printing, omitting print...\n");
                        continue;
                    case DPrintErrorEthernet:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: Ethernet core does not support TileSlice printing, omitting print...\n");
                        continue;
                    default:
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "Warning: TileSlice printing failed with unknown return code {}, omitting print...\n",
                            tile_slice.header.return_code);
                        continue;
                }

                // No error codes, print the TileSlice
                const uint8_t* data = tile_slice.data.data();
                uint32_t i = 0;
                bool count_exceeded = false;
                for (int h = tile_slice.header.slice_range.h0; h < tile_slice.header.slice_range.h1;
                     h += tile_slice.header.slice_range.hs) {
                    for (int w = tile_slice.header.slice_range.w0; w < tile_slice.header.slice_range.w1;
                         w += tile_slice.header.slice_range.ws) {
                        // If the number of data specified by the SliceRange exceeds the number that was
                        // saved in the print buffer (set by the MAX_COUNT template parameter in the
                        // TileSlice), then break early.
                        if (i >= tile_slice.header.data_count) {
                            count_exceeded = true;
                            break;
                        }
                        tt::DataFormat data_format = static_cast<tt::DataFormat>(tile_slice.header.data_format);
                        switch (data_format) {
                            case tt::DataFormat::Float16_b: {
                                const uint16_t* float16_b_ptr = reinterpret_cast<const uint16_t*>(data);
                                fmt::format_to(
                                    std::back_inserter(buffer.buffer), format, bfloat16_to_float(float16_b_ptr[i]));
                                break;
                            }
                            case tt::DataFormat::Float32: {
                                const float* float32_ptr = reinterpret_cast<const float*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, float32_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::Bfp4_b:
                            case tt::DataFormat::Bfp8_b: {
                                // Saved the exponent and data together
                                const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                                uint8_t val = (data_ptr[i] >> 8) & 0xFF;
                                uint8_t exponent = data_ptr[i] & 0xFF;
                                uint32_t bit_val = convert_bfp_to_u32(data_format, val, exponent, false);
                                fmt::format_to(
                                    std::back_inserter(buffer.buffer), format, *reinterpret_cast<float*>(&bit_val));
                                break;
                            }
                            case tt::DataFormat::Int8: {
                                const int8_t* data_ptr = reinterpret_cast<const int8_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt8: {
                                const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt16: {
                                const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::Int32: {
                                const int32_t* data_ptr = reinterpret_cast<const int32_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (int)data_ptr[i]);
                                break;
                            }
                            case tt::DataFormat::UInt32: {
                                const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(data);
                                fmt::format_to(std::back_inserter(buffer.buffer), format, (unsigned int)data_ptr[i]);
                                break;
                            }
                            default: break;
                        }
                        if (w + tile_slice.header.slice_range.ws < tile_slice.header.slice_range.w1) {
                            buffer.buffer.append(" "sv);
                        }
                        i++;
                    }

                    // Break outer loop as well if MAX COUNT exceeded, also print a message to let the user
                    // know that the slice has been truncated.
                    if (count_exceeded) {
                        fmt::format_to(
                            std::back_inserter(buffer.buffer),
                            "<TileSlice data truncated due to exceeding max count ({})>\n",
                            tile_slice.header.data_count);
                        break;
                    }

                    if (tile_slice.header.endl_rows) {
                        buffer.buffer.append("\n"sv);
                    }
                }

                break;
            }
            case 'A':  // dp_typed_array_t
            {
                const auto& arr = std::get<TypedArray>(buffer.argument_values[placeholder.arg_id]);
                std::string elements_format = "{" + std::string(placeholder.format_spec) + "} ";
                switch (arr.type) {
                    case tt::DataFormat::Float16:
                    case tt::DataFormat::Bfp8:
                    case tt::DataFormat::Bfp4:
                    case tt::DataFormat::Bfp2:
                    case tt::DataFormat::Lf8:
                    case tt::DataFormat::Bfp8_b:
                    case tt::DataFormat::Bfp4_b:
                    case tt::DataFormat::Bfp2_b:
                    case tt::DataFormat::Float16_b:
                    case tt::DataFormat::UInt16: elements_format = elements_format + elements_format; break;
                    case tt::DataFormat::Int8:
                    case tt::DataFormat::UInt8:
                        elements_format = elements_format + elements_format + elements_format + elements_format;
                        break;
                    default:
                    case tt::DataFormat::Tf32:
                    case tt::DataFormat::Float32:
                    case tt::DataFormat::UInt32:
                    case tt::DataFormat::Int32:
                        // Do nothing, single element format is sufficient
                        break;
                }
                format = fmt::runtime(elements_format);
                for (uint32_t datum : arr.data) {
                    switch (arr.type) {
                        case tt::DataFormat::Float16:
                        case tt::DataFormat::Bfp8:
                        case tt::DataFormat::Bfp4:
                        case tt::DataFormat::Bfp2:
                        case tt::DataFormat::Lf8:
                            fmt::format_to(
                                std::back_inserter(buffer.buffer),
                                format,
                                make_float(5, 10, datum & 0xffff),
                                make_float(5, 10, (datum >> 16) & 0xffff));
                            break;
                        case tt::DataFormat::Bfp8_b:
                        case tt::DataFormat::Bfp4_b:
                        case tt::DataFormat::Bfp2_b:
                        case tt::DataFormat::Float16_b:
                            fmt::format_to(
                                std::back_inserter(buffer.buffer),
                                format,
                                make_float(8, 7, datum & 0xffff),
                                make_float(8, 7, (datum >> 16) & 0xffff));
                            break;
                        case tt::DataFormat::Tf32:
                            fmt::format_to(std::back_inserter(buffer.buffer), format, make_float(8, 10, datum));
                            break;
                        case tt::DataFormat::Float32: {
                            float value;
                            memcpy(&value, &datum, sizeof(float));
                            fmt::format_to(std::back_inserter(buffer.buffer), format, value);
                        } break;
                        case tt::DataFormat::UInt32:
                            fmt::format_to(std::back_inserter(buffer.buffer), format, datum);
                            break;
                        case tt::DataFormat::UInt16:
                            fmt::format_to(std::back_inserter(buffer.buffer), format, datum & 0xffff, datum >> 16);
                            break;
                        case tt::DataFormat::Int32:
                            fmt::format_to(std::back_inserter(buffer.buffer), format, static_cast<int32_t>(datum));
                            break;
                        case tt::DataFormat::Int8:
                            fmt::format_to(
                                std::back_inserter(buffer.buffer),
                                format,
                                static_cast<int>(static_cast<int8_t>(datum & 0xff)),
                                static_cast<int>(static_cast<int8_t>((datum >> 8) & 0xff)),
                                static_cast<int>(static_cast<int8_t>((datum >> 16) & 0xff)),
                                static_cast<int>(static_cast<int8_t>((datum >> 24) & 0xff)));
                            break;
                        case tt::DataFormat::UInt8:
                            fmt::format_to(
                                std::back_inserter(buffer.buffer),
                                format,
                                static_cast<unsigned int>(datum & 0xff),
                                static_cast<unsigned int>((datum >> 8) & 0xff),
                                static_cast<unsigned int>((datum >> 16) & 0xff),
                                static_cast<unsigned int>((datum >> 24) & 0xff));
                            break;
                        default:
                            fmt::format_to(std::back_inserter(buffer.buffer), "Unknown data format {} ", arr.type);
                            break;
                    }
                }
                break;
            }
            case '/':  // Long type: '/' followed by sub-type character
            {
                TT_ASSERT(placeholder.enum_base_type_id != 0, "Unsupported long type in format placeholder");
                // Currently only '/e' (enum) and /E (flag enum) are supported
                // Get the integer value from the argument (using the base type)
                auto& arg = buffer.argument_values[placeholder.arg_id];
                std::visit(
                    [&placeholder, &buffer](auto&& v) {
                        using int_type = std::decay_t<decltype(v)>;
                        if constexpr (std::is_integral_v<int_type>) {
                            int_type int_val = static_cast<int_type>(v);

                            // Build the enum string representation, then format it with the user's format spec.
                            fmt::memory_buffer enum_str;
                            const auto* ei = placeholder.enum_info;

                            // Append "TypeName::" prefix when full name is requested
                            auto append_type_prefix = [&]() {
                                if (placeholder.enum_use_full_name) {
                                    enum_str.append(placeholder.enum_type_name);
                                    enum_str.append("::"sv);
                                }
                            };

                            // Format an unrecognized value as (TypeName)integer
                            auto append_raw_value = [&](int_type val) {
                                fmt::format_to(std::back_inserter(enum_str), "({}){}", placeholder.enum_type_name, val);
                            };

                            if (!ei || ei->enumerators.empty()) {
                                // No DWARF info available
                                append_raw_value(int_val);
                            } else if (placeholder.enum_is_flag && int_val != 0) {
                                // Print bitfield: Flag1 | Flag3 or BitEnum::Flag1 | BitEnum::Flag3
                                bool first = true;
                                int_type remaining = int_val;
                                for (const auto& [eval, ename] : ei->enumerators) {
                                    if (eval != 0 && (remaining & eval) == eval) {
                                        if (!first) {
                                            enum_str.append(" | "sv);
                                        }
                                        append_type_prefix();
                                        enum_str.append(std::string_view(ename));
                                        remaining &= ~eval;
                                        first = false;
                                    }
                                }
                                if (remaining != 0) {
                                    if (!first) {
                                        enum_str.append(" | "sv);
                                    }
                                    append_raw_value(remaining);
                                }
                            } else {
                                // Non-bitfield: find exact match
                                const std::string* found_name = nullptr;
                                for (const auto& [eval, ename] : ei->enumerators) {
                                    if (eval == int_val) {
                                        found_name = &ename;
                                        break;  // Use first match
                                    }
                                }
                                if (found_name) {
                                    append_type_prefix();
                                    enum_str.append(std::string_view(*found_name));
                                } else {
                                    append_raw_value(int_val);
                                }
                            }

                            // Apply user's format spec (alignment, width, fill, etc.)
                            auto enum_sv = std::string_view(enum_str.data(), enum_str.size());
                            fmt::format_to(
                                std::back_inserter(buffer.buffer), fmt::runtime(placeholder.fmt_format), enum_sv);
                        }
                    },
                    arg);
                break;
            }
            case 's':  // string pointer — resolve from .device_print_strings if possible, else hex
            {
                auto& arg = buffer.argument_values[placeholder.arg_id];
                uint64_t str_addr = std::get<pointer_t>(arg);
                if (str_addr >= format_strings_address &&
                    str_addr < format_strings_address + format_strings_bytes.size()) {
                    const char* str_ptr = reinterpret_cast<const char*>(
                        format_strings_bytes.data() + (str_addr - format_strings_address));
                    fmt::format_to(
                        std::back_inserter(buffer.buffer),
                        fmt::runtime(placeholder.fmt_format),
                        std::string_view(str_ptr));
                } else {
                    fmt::format_to(std::back_inserter(buffer.buffer), "0x{:x}", str_addr);
                }
                break;
            }
            case 'p':  // generic pointer — print as hex address
            {
                auto& arg = buffer.argument_values[placeholder.arg_id];
                uint64_t ptr_val = std::get<pointer_t>(arg);
                fmt::format_to(std::back_inserter(buffer.buffer), "0x{:x}", ptr_val);
                break;
            }
            case 'c': {
                const auto& info = std::get<TopCallstackInfo>(buffer.argument_values[placeholder.arg_id]);
                format_top_callstack(buffer.buffer, info);
                break;
            }
            default: TT_THROW("Unsupported type_id in format placeholder (format_message): {}", placeholder.type_id);
        }
    }
    auto& plain_text_part = string_info.plain_text_parts[string_info.placeholders.size()];
    buffer.buffer.append(plain_text_part.data(), plain_text_part.data() + plain_text_part.size());
    return std::string_view(buffer.buffer.data(), buffer.buffer.size());
}

template <uint8_t PointerSize>
DevicePrintParser::ArgumentValue DevicePrintParserImpl<PointerSize>::read_argument_from_payload(
    char type_id, std::span<const std::byte> payload_bytes, std::size_t& offset) const {
    switch (type_id) {
        case 'b':  // int8_t
            return read_value_from_payload<int8_t>(payload_bytes, offset);
        case 'B':  // uint8_t
            return read_value_from_payload<uint8_t>(payload_bytes, offset);
        case 'h':  // int16_t
            return read_value_from_payload<int16_t>(payload_bytes, offset);
        case 'H':  // uint16_t
            return read_value_from_payload<uint16_t>(payload_bytes, offset);
        case 'i':  // int32_t
            return read_value_from_payload<int32_t>(payload_bytes, offset);
        case 'I':  // uint32_t
            return read_value_from_payload<uint32_t>(payload_bytes, offset);
        case 'q':  // int64_t
            return read_value_from_payload<int64_t>(payload_bytes, offset);
        case 'Q':  // uint64_t
            return read_value_from_payload<uint64_t>(payload_bytes, offset);
        case 'f':  // float
            return read_value_from_payload<float>(payload_bytes, offset);
        case 'd':  // double
            return read_value_from_payload<double>(payload_bytes, offset);
        case '?':  // bool
            return read_value_from_payload<bool>(payload_bytes, offset);
        case 'e':  // bf4_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            uint8_t val = (data >> 8) & 0xFF;
            uint8_t exponent = data & 0xFF;
            uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp4_b, val, exponent, false);
            return *reinterpret_cast<float*>(&bit_val);
        }
        case 'E':  // bf8`_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            uint8_t val = (data >> 8) & 0xFF;
            uint8_t exponent = data & 0xFF;
            uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp8_b, val, exponent, false);
            return *reinterpret_cast<float*>(&bit_val);
        }
        case 'w':  // bf16_t, but stored as float
        {
            uint16_t data = read_value_from_payload<uint16_t>(payload_bytes, offset);
            auto value = bfloat16_to_float(data);
            return value;
        }
        case 't':  // TileSlice, but `pad` field carried info about MAX_BYTES
        {
            TileSliceDynamic tile_slice;
            tile_slice.header = read_value_from_payload<TileSliceHostDev<0>>(payload_bytes, offset);
            tile_slice.data.resize(tile_slice.header.pad);
            for (size_t i = 0; i < tile_slice.header.pad; ++i) {
                tile_slice.data[i] = read_value_from_payload<uint8_t>(payload_bytes, offset);
            }
            return tile_slice;
        }
        case 'A':  // dp_typed_array_t: [len, type, data[0..len-1]]
        {
            TypedArray arr;
            uint32_t packed_len_type = read_value_from_payload<uint32_t>(payload_bytes, offset);
            uint16_t len = static_cast<uint16_t>(packed_len_type >> 16);
            arr.type = static_cast<tt::DataFormat>(packed_len_type & 0xffff);
            arr.data.resize(len);
            for (uint32_t i = 0; i < len; ++i) {
                arr.data[i] = read_value_from_payload<uint32_t>(payload_bytes, offset);
            }
            return arr;
        }
        case 'c': {
            // The device encodes pc/ra/skip_frames as its own pointer width (pointer_t, selected from
            // the ELF). It marks an unknown pc/ra with the all-ones sentinel for that width, so on a
            // 32-bit device that sentinel is UINT32_MAX; widen it to the 64-bit sentinel that
            // TopCallstackInfo uses. On a 64-bit device pointer_t == uint64_t and this is a no-op.
            constexpr pointer_t sentinel = std::numeric_limits<pointer_t>::max();
            const pointer_t pc = read_value_from_payload<pointer_t>(payload_bytes, offset);
            const pointer_t ra = read_value_from_payload<pointer_t>(payload_bytes, offset);

            TopCallstackInfo info;
            info.pc = pc != sentinel ? pc : std::numeric_limits<uint64_t>::max();
            info.ra = ra != sentinel ? ra : std::numeric_limits<uint64_t>::max();
            info.skip_frames = read_value_from_payload<pointer_t>(payload_bytes, offset);
            return info;
        }
        case 's':  // string pointer (resolved from ELF section if possible, else hex)
        case 'p':  // generic pointer
            return read_value_from_payload<pointer_t>(payload_bytes, offset);
        default: TT_THROW("Unsupported type_id in format placeholder (read_argument_from_payload): {}", type_id);
    }
}

template <uint8_t PointerSize>
const typename DevicePrintParserImpl<PointerSize>::EnumInfo* DevicePrintParserImpl<PointerSize>::get_enum_info(
    std::string_view type_name) {
    if (auto it = enum_info_cache_.find(type_name); it != enum_info_cache_.end()) {
        return &it->second;
    }
    using ttexalens::native_elf::DwarfDieTag;
    const auto* dwarf_info = elf_file.get_dwarf_info();
    if (dwarf_info == nullptr) {
        // No DWARF info available (stripped binary, etc.)
        return nullptr;
    }

    // Resolve the (possibly qualified, e.g. "ns::Class::Enum") enum type name to
    // its DIE, then collect its DW_TAG_enumerator children as (value, name) pairs.
    auto enum_die = dwarf_info->get_die_by_name(type_name);
    if (!enum_die || enum_die->get_tag() != DwarfDieTag::enumeration_type) {
        return nullptr;
    }

    EnumInfo info;
    info.type_name = std::string(type_name);
    for (auto child = enum_die->get_first_child(); child; child = child->get_next_sibling()) {
        if (child->get_tag() != DwarfDieTag::enumerator) {
            continue;
        }
        std::string_view enumerator_name = child->get_name();
        if (enumerator_name.empty()) {
            continue;
        }
        // ConstantValue is variant<monostate, bool, int64_t, uint64_t, float, double>;
        // an enumerator carries an integral (or bool) value — anything else is skipped.
        std::optional<int64_t> enum_val = std::visit(
            [](auto&& v) -> std::optional<int64_t> {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, bool>) {
                    return static_cast<int64_t>(v);
                } else {
                    return std::nullopt;
                }
            },
            child->get_constant_value());
        if (enum_val) {
            info.enumerators.emplace_back(*enum_val, std::string(enumerator_name));
        }
    }

    if (info.enumerators.empty()) {
        return nullptr;
    }
    auto inserted = enum_info_cache_.emplace(std::string(type_name), std::move(info));
    return &inserted.first->second;
}

template <uint8_t PointerSize>
auto DevicePrintParserImpl<PointerSize>::resolve_top_callstack(const TopCallstackInfo& info) {
    using ttexalens::native_elf::CallstackEntry;

    std::vector<CallstackEntry> callstack;
    callstack.push_back({});

    if (elf_file.get_dwarf_info() == nullptr) {
        return callstack;
    }

    // Device sends offset from start of .text
    // We also need the section offset for DWARF lookup
    const uint64_t text_start = elf_file.get_code_load_address();

    const std::vector<ttexalens::native_elf::ElfFile> elfs = {elf_file};

    // The device leaves the UINT64 MAX sentinel when PC/RA wasn't captured
    const auto is_invalid_address = [](uint64_t addr) { return addr == std::numeric_limits<uint64_t>::max(); };

    // Unwinding past kernel boundary is currently unsupported
    // Stop if you reach a known terminal frame
    const auto is_terminal = [](const CallstackEntry& entry) {
        const auto& f = entry.function_name;
        return f == "kernel_main" || f == "run_kernel" || f == "main" || f == "_start";
    };

    // Resolves inline callstack with given text offset
    // Trims anything past first terminal
    const auto resolve = [&](uint64_t offset) -> std::vector<CallstackEntry> {
        const uint64_t address = text_start + offset;
        return ttexalens::native_elf::get_frame_callstack(elfs, address, /*extract_variables=*/false);
    };

    if (is_invalid_address(info.pc) || is_invalid_address(info.ra)) {
        return callstack;
    }

    auto pc_frames = resolve(info.pc);
    callstack.assign(std::make_move_iterator(pc_frames.begin()), std::make_move_iterator(pc_frames.end()));

    auto ra_frames = resolve(info.ra);
    callstack.insert(
        callstack.end(), std::make_move_iterator(ra_frames.begin()), std::make_move_iterator(ra_frames.end()));

    // Continuation sentinel, if we didn't terminate
    callstack.push_back({});

    // Slice [skip_frames, first_terminal]
    const auto terminal = std::find_if(callstack.begin(), callstack.end(), is_terminal);
    const auto end = terminal == callstack.end() ? terminal : std::next(terminal);
    const auto begin = callstack.begin() + std::min(info.skip_frames, callstack.size());

    // If the terminal is skipped, we have no valid frames to show
    if (begin >= end) {
        return std::vector<CallstackEntry>{{}};
    }

    callstack.erase(end, callstack.end());
    callstack.erase(callstack.begin(), begin);
    return callstack;
}

template <uint8_t PointerSize>
void DevicePrintParserImpl<PointerSize>::format_top_callstack(fmt::memory_buffer& out, const TopCallstackInfo& info) {
    auto sink = std::back_inserter(out);

    // Renders the resolved callstack as a tree, innermost frame first. Every real frame
    // contributes two lines — the function name, then its source location:
    //   "│  ├──┬ FUNC\r"
    //   "│  │  └ FILE:LINE\r"
    // The outermost real frame uses the └ connector and drops the trailing │. When the stack
    // couldn't be fully unwound, resolve_top_callstack appends a sentinel frame (always last),
    // which renders as a closing leaf:
    //   "│  └ ...\r"
    const auto stack = resolve_top_callstack(info);

    for (size_t idx = 0; idx < stack.size(); ++idx) {
        const auto& frame = stack[idx];
        const bool last = (idx + 1 == stack.size());

        if (last && !frame.function_name) {
            fmt::format_to(sink, "│  └ ...\r");
            break;
        }

        fmt::format_to(sink, "│  {}──┬ {}\r", last ? "└" : "├", frame.function_name.value_or("<unknown>"));
        // The file:line continuation column drops the │ on the last (outermost) frame.
        const char* cont = last ? "   " : "│  ";
        if (!frame.file_info) {
            fmt::format_to(sink, "│  {}└ <unknown>\r", cont);
        } else {
            fmt::format_to(sink, "│  {}└ {}:{}\r", cont, frame.file_info->file, frame.file_info->line);
        }
    }

    fmt::format_to(sink, "│");
}

}  // namespace tt::tt_metal
