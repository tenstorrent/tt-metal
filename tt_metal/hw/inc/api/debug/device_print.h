// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>

#include "hostdev/device_print_common.h"
#include "hostdev/device_print_structures.h"
#include "waypoint.h"
#include "internal/debug/dprint_buffer.h"
#include "noc_overlay_parameters.h"
#include "risc_common.h"
#include "stream_io_map.h"

#if defined(KERNEL_BUILD)
#include "dprint_tile.h"
#endif

#define DEVICE_PRINT_STRINGS_SECTION_NAME ".device_print_strings"
#define DEVICE_PRINT_STRINGS_INFO_SECTION_NAME ".device_print_strings_info"

// Start of the .device_print_strings_info section, which represents list of DevicePrintStringInfo structures.
extern char __device_print_strings_info_start[];

struct bf4_t {
    union {
        struct {
            uint8_t exponent;
            uint8_t mantissa;
        };
        uint16_t val;
    };
    bf4_t(uint16_t val) : val(val) {}
    bf4_t(uint8_t exponent, uint8_t mantissa) : exponent(exponent), mantissa(mantissa) {}
};

struct bf8_t {
    union {
        struct {
            uint8_t exponent;
            uint8_t mantissa;
        };
        uint16_t val;
    };
    bf8_t(uint16_t val) : val(val) {}
    bf8_t(uint8_t exponent, uint8_t mantissa) : exponent(exponent), mantissa(mantissa) {}
};

struct bf16_t {
    uint16_t val;
    bf16_t(uint16_t val) : val(val) {}
};

#ifdef UCK_CHLKC_UNPACK
#define DEVICE_PRINT_UNPACK(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_UNPACK(format, ...)
#endif

#ifdef UCK_CHLKC_MATH
#define DEVICE_PRINT_MATH(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_MATH(format, ...)
#endif

#ifdef UCK_CHLKC_PACK
#define DEVICE_PRINT_PACK(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_PACK(format, ...)
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_DM)
#define DEVICE_PRINT_DATA0(format, ...)      \
    if (noc_index == 0) {                    \
        DEVICE_PRINT(format, ##__VA_ARGS__); \
    }
#define DEVICE_PRINT_DATA1(format, ...)      \
    if (noc_index == 1) {                    \
        DEVICE_PRINT(format, ##__VA_ARGS__); \
    }
#else
#define DEVICE_PRINT_DATA0(format, ...)
#define DEVICE_PRINT_DATA1(format, ...)
#endif

#if defined(KERNEL_BUILD)
#define DEVICE_PRINT_IS_KERNEL 1
#else
#define DEVICE_PRINT_IS_KERNEL 0
#endif

#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF) && defined(USE_DEVICE_PRINT)
#define DEVICE_PRINT_GET_STRING_INFO_ADDRESS(variable_name, updated_format)                                    \
    std::uintptr_t variable_name = 0;                                                                          \
    {                                                                                                          \
        static const auto allocated_string __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = \
            updated_format.to_array();                                                                         \
        static const auto allocated_file_string                                                                \
            __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = []() {                         \
                device_print_detail::helpers::static_string<sizeof(__FILE__)> file_str;                        \
                for (std::size_t i = 0; i < sizeof(__FILE__); ++i) {                                           \
                    file_str.push_back(__FILE__[i]);                                                           \
                }                                                                                              \
                return file_str.to_array();                                                                    \
            }();                                                                                               \
        static device_print_detail::structures::DevicePrintStringInfo allocated_string_info                    \
            __attribute__((section(DEVICE_PRINT_STRINGS_INFO_SECTION_NAME), used)) = {                         \
                allocated_string.data(), allocated_file_string.data(), __LINE__};                              \
        variable_name = reinterpret_cast<std::uintptr_t>(&allocated_string_info);                              \
    }

#define DEVICE_PRINT(format, ...)                                                                                     \
    {                                                                                                                 \
        auto device_print_info_address = ([](auto&&... _device_print_args_) __attribute__((always_inline)) {          \
            /* Validate format string syntax */                                                                       \
            static_assert(                                                                                            \
                device_print_detail::checks::is_valid_format_string(format),                                          \
                "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");                     \
            /* Validate placeholder format */                                                                         \
            static_assert(                                                                                            \
                !device_print_detail::checks::has_mixed_placeholders(format),                                         \
                "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");              \
            /* For indexed placeholders, validate no index exceeds argument count */                                  \
            static_assert(                                                                                            \
                !device_print_detail::checks::has_indexed_placeholders(format) ||                                     \
                    device_print_detail::checks::get_max_index(format) <                                              \
                        device_print_detail::helpers::count_arguments(_device_print_args_...),                        \
                "Placeholder index exceeds number of arguments");                                                     \
            /* For indexed placeholders, validate all arguments are referenced */                                     \
            static_assert(                                                                                            \
                !device_print_detail::checks::has_indexed_placeholders(format) ||                                     \
                    device_print_detail::checks::all_arguments_referenced(format, _device_print_args_...),            \
                "All arguments must be referenced when using indexed placeholders");                                  \
            /* For non-indexed placeholders, count must match argument count */                                       \
            static_assert(                                                                                            \
                device_print_detail::checks::has_indexed_placeholders(format) ||                                      \
                    device_print_detail::checks::count_placeholders(format) ==                                        \
                        device_print_detail::helpers::count_arguments(_device_print_args_...),                        \
                "Number of {} placeholders must match number of arguments");                                          \
            /* Update format to include all necessary data */                                                         \
            constexpr auto updated_format =                                                                           \
                device_print_detail::formatting::update_format_string_from_args(format, _device_print_args_...);      \
            /* Store updated format string in a special section for device_print */                                   \
            DEVICE_PRINT_GET_STRING_INFO_ADDRESS(device_print_info_address, updated_format);                          \
            return device_print_info_address;                                                                         \
        }(__VA_ARGS__));                                                                                              \
        auto header = ([](auto&&... _device_print_args_) __attribute__((always_inline)) {                             \
            /* Generate device_print message header */                                                                \
            constexpr auto message_size =                                                                             \
                device_print_detail::serialization::get_total_message_size(_device_print_args_...);                   \
            device_print_detail::structures::DevicePrintHeader header = {};                                           \
            header.is_kernel = DEVICE_PRINT_IS_KERNEL;                                                                \
            header.risc_id = PROCESSOR_INDEX;                                                                         \
            header.message_payload = message_size - sizeof(header); /* Payload size does not include header itself */ \
            return header;                                                                                            \
        }(__VA_ARGS__));                                                                                              \
        /* Get device_print buffer*/                                                                                  \
        volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer = get_device_print_buffer();                  \
        /* Get buffer lock, since we are using a single buffer per L1 instead of per risc */                          \
        /* Check if we have enough space in the buffer or we need to wrap buffer */                                   \
        /* Wait for enough space in the buffer (if reader needs to catch up). */                                      \
        /* Update message header with string info index */                                                            \
        /* Serialize message header */                                                                                \
        auto write_position = device_print_detail::begin_message_write(header, device_print_info_address);            \
        /* Serialize arguments */                                                                                     \
        auto device_print_buffer_ptr = &(device_print_buffer->data[0]) + write_position;                              \
        device_print_detail::serialization::serialize_arguments(device_print_buffer_ptr, ##__VA_ARGS__);              \
        /* Update write pointer and release buffer lock */                                                            \
        device_print_detail::end_message_write();                                                                     \
    }

#define DEVICE_PRINT_INITIALIZE_LOCK() device_print_detail::locking::initialize_lock()
#define DEVICE_PRINT_KERNEL_FINISHED() device_print_detail::locking::update_kernel_finished()

namespace device_print_detail {

template <typename BufferType>
using device_print_buffer_ptr = volatile tt_l1_ptr BufferType*;

namespace helpers {

template <typename... Args>
constexpr std::size_t count_arguments(const Args&...) {
    return sizeof...(Args);
}

// Helper to check if a character is a digit
constexpr bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Compile-time string class for building the result
template <std::size_t N>
struct static_string {
    char data[N + 1];
    std::size_t size;

    constexpr static_string() : data{}, size(0) {}

    constexpr void push_back(char c) {
        if (size < N) {
            data[size++] = c;
            data[size] = '\0';  // Ensure null termination
        }
    }

    constexpr void push_back_uint32(uint32_t value) {
        // Special case for zero
        if (value == 0) {
            push_back('0');
            return;
        }

        int digits = 0;
        uint64_t digit_reader = 1;

        while (digit_reader <= value) {
            digit_reader *= 10;
            digits++;
        }
        digit_reader /= 10;

        for (int i = 0; i < digits; ++i) {
            char digit_char = '0' + (value / digit_reader) % 10;
            push_back(digit_char);
            digit_reader /= 10;
        }
    }

    constexpr std::array<char, N + 1> to_array() const {
        std::array<char, N + 1> arr = {};
        for (std::size_t i = 0; i < size; ++i) {
            arr[i] = data[i];
        }
        arr[size] = '\0';
        return arr;
    }

    // Helper to create a compact array of the actual used size
    template <std::size_t... Is>
    constexpr std::array<char, sizeof...(Is)> to_compact_array_impl(std::index_sequence<Is...>) const {
        return {{data[Is]...}};
    }

    // Returns an array sized exactly to fit the string content (size + 1 for null terminator)
    constexpr auto to_compact_array() const { return to_compact_array_impl(std::make_index_sequence<size + 1>{}); }

    template <std::size_t M>
    constexpr bool check(const char (&expected)[M]) const {
        if (size != M - 1) {
            return false;
        }
        for (std::size_t i = 0; i < size; ++i) {
            if (data[i] != expected[i]) {
                return false;
            }
        }
        return true;
    }

    constexpr const char* c_str() const { return data; }
};

}  // namespace helpers

namespace parsing {

// Helper struct to return both parsed value and new position
struct IndexParseResult {
    uint32_t value;
    std::size_t new_pos;
};

// Helper to parse an integer from format string starting at position i
// Returns the parsed value and the new position after the digits
constexpr IndexParseResult parse_index(const char* format, std::size_t i, std::size_t format_len) {
    uint32_t value = 0;
    std::size_t pos = i;
    while (pos < format_len && helpers::is_digit(format[pos])) {
        value = value * 10 + (format[pos] - '0');
        ++pos;
    }
    return IndexParseResult{value, pos};
}

// Token types for format string parsing
enum class TokenType {
    Placeholder,         // {} or {N}
    EscapedOpenBrace,    // {{
    EscapedCloseBrace,   // }}
    InvalidPlaceholder,  // Invalid { sequence
    RegularChar          // Any other character
};

// Result of parsing a single token from format string
struct FormatToken {
    TokenType type = TokenType::RegularChar;
    std::size_t end_pos = 0;  // Position to continue parsing from (after this token)

    // Placeholder-specific data
    std::optional<uint32_t> index;    // The N in {N}, or -1 for {} or escape sequences
    std::optional<char> fill;         // <a character other than '{' or '}'>
    std::optional<char> align;        // '<' | '>' | '^'
    std::optional<char> sign;         // '+' | '-' | ' ']
    bool use_alternate_form = false;  // '#'
    bool use_zero_padding = false;    // '0'
    std::optional<uint32_t> width;
    std::optional<uint32_t> precision;
    bool use_current_locale = false;  // 'L'
    std::optional<char> format_type;  // 'a' | 'A' | 'b' | 'B' | 'c' | 'd' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'o' |
                                      // 'p' | 's' | 'x' | 'X' | '?'

    constexpr bool is_indexed() const { return index.has_value(); }
};

// Parse a single token from the format string at position i
// This is the main tokenizer that handles all format string elements:
// - Placeholders: {} and {N}
// - Escape sequences: {{ and }}
// - Regular characters
template <std::size_t N>
constexpr FormatToken parse_format_token(const char (&format)[N], std::size_t i) {
    constexpr std::size_t format_len = N - 1;

    if (i >= format_len) {
        return FormatToken{TokenType::RegularChar, i + 1};
    }

    char c = format[i];

    // Check for escaped opening brace {{
    if (c == '{' && i + 1 < format_len && format[i + 1] == '{') {
        return FormatToken{TokenType::EscapedOpenBrace, i + 2};
    }

    // Check for escaped closing brace }}
    if (c == '}' && i + 1 < format_len && format[i + 1] == '}') {
        return FormatToken{TokenType::EscapedCloseBrace, i + 2};
    }

    // Check for placeholder
    if (c == '{') {
        i++;
        if (i >= format_len) {
            // '{' at end of string is invalid
            return FormatToken{TokenType::InvalidPlaceholder, i};
        }

        // We are trying to mimic fmtlib format specifiers here:
        // replacement_field ::= "{" [arg_id] [":" (format_spec | chrono_format_spec)] "}"
        // arg_id            ::= integer | identifier
        // integer           ::= digit+
        // digit             ::= "0"..."9"
        // identifier        ::= id_start id_continue*
        // id_start          ::= "a"..."z" | "A"..."Z" | "_"
        // id_continue       ::= id_start | digit
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

        // Initially we mark parsed_token as invalid for simpler code and update it as Placeholder if valid
        FormatToken parsed_token{TokenType::InvalidPlaceholder};

        // arg_id parsing
        if (helpers::is_digit(format[i])) {
            IndexParseResult result = parse_index(format, i, format_len);
            i = result.new_pos;
            parsed_token.index = result.value;
        }

        // Check for format specifier
        if (i < format_len && format[i] == ':') {
            i++;  // Skip ':'

            // Parse fill and align
            if (i < format_len && (format[i] == '<' || format[i] == '>' || format[i] == '^')) {
                parsed_token.align = format[i];
                i++;
            } else if (
                i + 1 < format_len && format[i] != '{' && format[i] != '}' &&
                (format[i + 1] == '<' || format[i + 1] == '>' || format[i + 1] == '^')) {
                parsed_token.fill = format[i];
                parsed_token.align = format[i + 1];
                i += 2;
            }

            // Parse sign
            if (i < format_len && (format[i] == '+' || format[i] == '-' || format[i] == ' ')) {
                parsed_token.sign = format[i];
                i++;
            }

            // Parse alternate form
            if (i < format_len && format[i] == '#') {
                parsed_token.use_alternate_form = true;
                i++;
            }

            // Parse zero padding
            if (i < format_len && format[i] == '0') {
                parsed_token.use_zero_padding = true;
                i++;
            }

            // Parse width
            if (i < format_len && helpers::is_digit(format[i])) {
                IndexParseResult result = parse_index(format, i, format_len);
                i = result.new_pos;
                parsed_token.width = result.value;
            }

            // Parse precision
            if (i + 1 < format_len && format[i] == '.' && helpers::is_digit(format[i + 1])) {
                i++;  // Skip '.'
                IndexParseResult result = parse_index(format, i, format_len);
                i = result.new_pos;
                parsed_token.precision = result.value;
            }

            // Parse locale option
            if (i < format_len && format[i] == 'L') {
                parsed_token.use_current_locale = true;
                i++;
            }

            // Parse format type
            if (i < format_len &&
                (format[i] == 'a' || format[i] == 'A' || format[i] == 'b' || format[i] == 'B' || format[i] == 'c' ||
                 format[i] == 'd' || format[i] == 'e' || format[i] == 'E' || format[i] == 'f' || format[i] == 'F' ||
                 format[i] == 'g' || format[i] == 'G' || format[i] == 'o' || format[i] == 'p' || format[i] == 's' ||
                 format[i] == 'x' || format[i] == 'X' || format[i] == '?')) {
                parsed_token.format_type = format[i];
                i++;
            }
        }

        // Check for closing brace
        if (i < format_len && format[i] == '}') {
            parsed_token.type = TokenType::Placeholder;
            parsed_token.end_pos = i + 1;
            return parsed_token;
        }

        parsed_token.end_pos = i;
        return parsed_token;
    }

    // Regular character
    return FormatToken{TokenType::RegularChar, i + 1};
}

}  // namespace parsing

namespace checks {

// Helper to validate format string for invalid brace sequences
// Returns true if format string is valid, false if it contains errors
template <std::size_t N>
constexpr bool is_valid_format_string(const char (&format)[N]) {
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);

        // Check for invalid placeholder syntax
        if (token.type == parsing::TokenType::InvalidPlaceholder) {
            return false;
        }

        i = token.end_pos;
    }
    return true;
}

// Helper to check for mixed placeholder styles (both {} and {N})
// This should fail validation per fmtlib rules
template <std::size_t N>
constexpr bool has_mixed_placeholders(const char (&format)[N]) {
    bool found_indexed = false;
    bool found_unindexed = false;

    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder) {
            if (token.is_indexed()) {
                found_indexed = true;
            } else {
                found_unindexed = true;
            }
            if (found_indexed && found_unindexed) {
                return true;
            }
        }
        i = token.end_pos;
    }
    return false;
}

// Helper to detect if format string uses indexed placeholders ({0}, {1}, etc.)
// Returns true if ANY placeholder has an index
template <std::size_t N>
constexpr bool has_indexed_placeholders(const char (&format)[N]) {
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            return true;
        }
        i = token.end_pos;
    }
    return false;
}

// Helper to find the maximum index used in format string
template <std::size_t N>
constexpr uint32_t get_max_index(const char (&format)[N]) {
    uint32_t max_index = 0;
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            max_index = std::max(max_index, token.index.value());
        }
        i = token.end_pos;
    }
    return max_index;
}

// Helper to validate that all arguments are referenced in indexed format
// Returns true if all argument indices from 0 to arg_count-1 are used at least once
template <std::size_t N, typename... Args>
constexpr bool all_arguments_referenced(const char (&format)[N], const Args&... args) {
    if (sizeof...(args) == 0) {
        return true;
    }

    // Track which arguments are referenced
    std::array<bool, sizeof...(Args)> referenced = {};

    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            if (static_cast<std::size_t>(token.index.value()) < sizeof...(args)) {
                referenced[token.index.value()] = true;
            }
        }
        i = token.end_pos;
    }

    // Check that all arguments from 0 to arg_count-1 are referenced
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
        if (!referenced[i]) {
            return false;
        }
    }
    return true;
}

// Helper to count placeholders in format string at compile time
template <std::size_t N>
constexpr std::size_t count_placeholders(const char (&format)[N]) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder) {
            ++count;
        }
        i = token.end_pos;
    }
    return count;
}

}  // namespace checks

namespace formatting {

struct device_print_type_info {
    char type_char;
    uint32_t size_in_bytes;
};

// Type-to-info mapping for format strings and serialization
template <typename T>
struct device_print_type {
    static constexpr device_print_type_info value = {'#', 0};  // Unknown type default
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, T argument) {
        static_assert(!std::is_same_v<T, T>, "No serialization defined for this type");
    }
};

// Specializations for different types
template <>
struct device_print_type<std::int8_t> {
    static constexpr device_print_type_info value = {'b', sizeof(std::int8_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, int8_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<int8_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::uint8_t> {
    static constexpr device_print_type_info value = {'B', sizeof(std::uint8_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, uint8_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint8_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::int16_t> {
    static constexpr device_print_type_info value = {'h', sizeof(std::int16_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, int16_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<int16_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::uint16_t> {
    static constexpr device_print_type_info value = {'H', sizeof(std::uint16_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, uint16_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint16_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::int32_t> {
    static constexpr device_print_type_info value = {'i', sizeof(std::int32_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, int32_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<int32_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::uint32_t> {
    static constexpr device_print_type_info value = {'I', sizeof(std::uint32_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, uint32_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::int64_t> {
    static constexpr device_print_type_info value = {'q', sizeof(std::int64_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, int64_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<int64_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<std::uint64_t> {
    static constexpr device_print_type_info value = {'Q', sizeof(std::uint64_t)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, uint64_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint64_t>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<float> {
    static constexpr device_print_type_info value = {'f', sizeof(float)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, float argument) {
        *reinterpret_cast<device_print_buffer_ptr<float>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<double> {
    static constexpr device_print_type_info value = {'d', sizeof(double)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, double argument) {
        *reinterpret_cast<device_print_buffer_ptr<double>>(device_print_buffer + offset) = argument;
    }
};
template <>
struct device_print_type<bool> {
    static constexpr device_print_type_info value = {'?', 1};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, bool argument) {
        *(device_print_buffer + offset) = static_cast<uint8_t>(argument);
    }
};
template <>
struct device_print_type<bf4_t> {
    static constexpr device_print_type_info value = {'e', sizeof(uint16_t)};
    static_assert(sizeof(bf4_t) == sizeof(uint16_t), "bf4_t must be 16 bits");
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, bf4_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint16_t>>(device_print_buffer + offset) =
            static_cast<uint16_t>(argument.val);
    }
};
template <>
struct device_print_type<bf8_t> {
    static constexpr device_print_type_info value = {'E', sizeof(uint16_t)};
    static_assert(sizeof(bf8_t) == sizeof(uint16_t), "bf8_t must be 16 bits");
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, bf8_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint16_t>>(device_print_buffer + offset) =
            static_cast<uint16_t>(argument.val);
    }
};
template <>
struct device_print_type<bf16_t> {
    static constexpr device_print_type_info value = {'w', sizeof(uint16_t)};
    static_assert(sizeof(bf16_t) == sizeof(uint16_t), "bf16_t must be 16 bits");
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, bf16_t argument) {
        *reinterpret_cast<device_print_buffer_ptr<uint16_t>>(device_print_buffer + offset) =
            static_cast<uint16_t>(argument.val);
    }
};

// Pointer types (including strings)
template <typename T>
struct device_print_type<T*> {
    static constexpr device_print_type_info value = {'p', sizeof(T*)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, T* argument) {
        if constexpr (sizeof(T*) == 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(T*) == 8) {
            *reinterpret_cast<device_print_buffer_ptr<uint64_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(T*) == 4 || sizeof(T*) == 8, "Unsupported pointer size");
        }
    }
};
template <>
struct device_print_type<char*> {
    static constexpr device_print_type_info value = {'s', sizeof(char*)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, char* argument) {
        if constexpr (sizeof(char*) == 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(char*) == 8) {
            *reinterpret_cast<device_print_buffer_ptr<uint64_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(char*) == 4 || sizeof(char*) == 8, "Unsupported pointer size");
        }
    }
};
template <>
struct device_print_type<const char*> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
    static void serialize(device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, const char* argument) {
        if constexpr (sizeof(const char*) == 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(const char*) == 8) {
            *reinterpret_cast<device_print_buffer_ptr<uint64_t>>(device_print_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(const char*) == 4 || sizeof(const char*) == 8, "Unsupported pointer size");
        }
    }
};

// Array types (treat as strings)
template <std::size_t N>
struct device_print_type<char[N]> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
};
template <std::size_t N>
struct device_print_type<const char[N]> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
};

#if defined(KERNEL_BUILD)
// TileSlice types
template <uint32_t MAX_BYTES>
struct device_print_type<TileSlice<MAX_BYTES>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<MAX_BYTES>)};
    static void serialize(
        device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, TileSlice<MAX_BYTES> argument) {
        static_assert(
            sizeof(TileSlice<MAX_BYTES>) % 4 == 0,
            "TileSlice size must be a multiple of 4 bytes for proper serialization");
        static_assert(MAX_BYTES < 256, "MAX_BYTES must be less than 256 to fit size in a single byte");
        argument.pad = MAX_BYTES;  // Store the actual size in the padding field
        TileSlice<MAX_BYTES>* argument_pointer = &argument;
        uint32_t* argument_as_uint32_ptr = reinterpret_cast<uint32_t*>(argument_pointer);
        for (uint32_t i = 0; i < sizeof(TileSlice<MAX_BYTES>); i += 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset + i) =
                argument_as_uint32_ptr[i / 4];
        }
    }
};
template <>
struct device_print_type<TileSlice<64>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<64>)};
    static void serialize(
        device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, TileSlice<64> argument) {
        static_assert(
            sizeof(TileSlice<64>) % 4 == 0, "TileSlice size must be a multiple of 4 bytes for proper serialization");
        argument.pad = 64;  // Store the actual size in the padding field
        TileSlice<64>* argument_pointer = &argument;
        uint32_t* argument_as_uint32_ptr = reinterpret_cast<uint32_t*>(argument_pointer);
        for (uint32_t i = 0; i < sizeof(TileSlice<64>); i += 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset + i) =
                argument_as_uint32_ptr[i / 4];
        }
    }
};
template <>
struct device_print_type<TileSlice<128>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<128>)};
    static void serialize(
        device_print_buffer_ptr<uint8_t> device_print_buffer, uint32_t offset, TileSlice<128> argument) {
        static_assert(
            sizeof(TileSlice<128>) % 4 == 0, "TileSlice size must be a multiple of 4 bytes for proper serialization");
        argument.pad = 128;  // Store the actual size in the padding field
        TileSlice<128>* argument_pointer = &argument;
        uint32_t* argument_as_uint32_ptr = reinterpret_cast<uint32_t*>(argument_pointer);
        for (uint32_t i = 0; i < sizeof(TileSlice<128>); i += 4) {
            *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer + offset + i) =
                argument_as_uint32_ptr[i / 4];
        }
    }
};
#endif

// Helper to get type character for a single type, removing cv-qualifiers and references
template <typename T>
constexpr device_print_type_info get_type_info() {
    using base_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return device_print_type<base_type>::value;
}

template <typename... Args>
constexpr std::array<device_print_type_info, sizeof...(Args)> get_types_info() {
    return {get_type_info<Args>()...};
}

template <typename... Args>
constexpr std::array<uint32_t, sizeof...(Args)> get_arg_reorder() {
    constexpr auto type_infos = get_types_info<Args...>();

    // Initialize to default ordering of arguments (0, 1, 2, ...).
    std::array<uint32_t, sizeof...(Args)> arg_reorder = {};
    for (std::size_t i = 0; i < arg_reorder.size(); ++i) {
        arg_reorder[i] = i;
    }

    // Sort arguments by size_in_bytes descending to optimize serialization (largest first)
    for (std::size_t i = 0; i < arg_reorder.size(); ++i) {
        for (std::size_t j = i + 1; j < arg_reorder.size(); ++j) {
            if (type_infos[arg_reorder[j]].size_in_bytes > type_infos[arg_reorder[i]].size_in_bytes) {
                uint32_t temp = arg_reorder[i];
                arg_reorder[i] = arg_reorder[j];
                arg_reorder[j] = temp;
            }
        }
    }
    return arg_reorder;
}

template <typename... Args>
constexpr std::array<uint32_t, sizeof...(Args)> get_arg_offsets() {
    constexpr auto type_infos = get_types_info<Args...>();
    constexpr auto arg_reorder = get_arg_reorder<Args...>();
    std::array<uint32_t, sizeof...(Args)> arg_memory_offsets = {};
    uint32_t current_offset = sizeof(structures::DevicePrintHeader::value);
    for (std::size_t i = 0; i < arg_memory_offsets.size(); ++i) {
        arg_memory_offsets[i] = current_offset;
        current_offset += type_infos[arg_reorder[i]].size_in_bytes;
    }
    std::array<uint32_t, sizeof...(Args)> arg_offset = {};
    for (std::size_t i = 0; i < arg_offset.size(); ++i) {
        arg_offset[arg_reorder[i]] = arg_memory_offsets[i];
    }
    return arg_offset;
}

// Main function to update format string with type information
// Supports both {} and {N} placeholder styles (fmtlib-compatible)
template <std::size_t N, typename... Args>
constexpr auto update_format_string(const char (&format)[N]) {
    constexpr std::size_t format_len = N - 1;  // Exclude null terminator

    // Calculate maximum result length.
    // Each {} placeholder (2 chars) expands to {N,T} where N is the arg index and T is the type char.
    // The net extra chars per placeholder = 2 + digits(max_index), where:
    //   - 1 digit  (N < 10):   net = 3
    //   - 2 digits (N < 100):  net = 4
    //   - 3 digits (N < 1000): net = 5
    // Use sizeof...(Args) to pick the right bound rather than always assuming 2.
    constexpr std::size_t num_args_ = sizeof...(Args);
    constexpr std::size_t max_index_digits_ = (num_args_ <= 9) ? 1 : (num_args_ <= 99) ? 2 : (num_args_ <= 999) ? 3 : 4;
    constexpr std::size_t result_len = format_len + (format_len / 2 + 1) * (2 + max_index_digits_);

    helpers::static_string<result_len> result;

    constexpr auto type_infos = get_types_info<Args...>();
    constexpr auto arg_reorder = get_arg_reorder<Args...>();

    std::size_t type_index = 0;

    for (std::size_t i = 0; i < format_len;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);

        if (token.type == parsing::TokenType::EscapedOpenBrace) {
            // Preserve escaped opening brace: {{
            result.push_back('{');
            result.push_back('{');
        } else if (token.type == parsing::TokenType::EscapedCloseBrace) {
            // Preserve escaped closing brace: }}
            result.push_back('}');
            result.push_back('}');
        } else if (token.type == parsing::TokenType::Placeholder) {
            // Determine the argument index for this placeholder
            uint32_t arg_index = token.is_indexed() ? token.index.value() : type_index++;

            // Unified handling for both indexed and non-indexed: output {index,type:format}
            result.push_back('{');

            // Output the index (updated with reordered index)
            result.push_back_uint32(arg_reorder[arg_index]);

            // Add comma and type character
            result.push_back(',');
            result.push_back(type_infos[arg_index].type_char);

            // If there are any format specifiers, add them
            bool has_format_spec = token.fill.has_value() || token.align.has_value() || token.sign.has_value() ||
                                   token.use_alternate_form || token.use_zero_padding || token.width.has_value() ||
                                   token.precision.has_value() || token.use_current_locale ||
                                   token.format_type.has_value();
            if (has_format_spec) {
                result.push_back(':');
                if (token.fill.has_value()) {
                    result.push_back(token.fill.value());
                }
                if (token.align.has_value()) {
                    result.push_back(token.align.value());
                }
                if (token.sign.has_value()) {
                    result.push_back(token.sign.value());
                }
                if (token.use_alternate_form) {
                    result.push_back('#');
                }
                if (token.use_zero_padding) {
                    result.push_back('0');
                }
                if (token.width.has_value()) {
                    result.push_back_uint32(token.width.value());
                }
                if (token.precision.has_value()) {
                    result.push_back('.');
                    result.push_back_uint32(token.precision.value());
                }
                if (token.use_current_locale) {
                    result.push_back('L');
                }
                if (token.format_type.has_value()) {
                    result.push_back(token.format_type.value());
                }
            }
            result.push_back('}');
        } else {
            // Regular character
            result.push_back(format[i]);
        }

        i = token.end_pos;
    }

    return result;
}

// New function that returns constexpr updated format string from arguments
// This allows calling with actual arguments: update_format_string_from_args("format {}", arg1, arg2, ...)
template <std::size_t N, typename... Args>
constexpr auto update_format_string_from_args(const char (&format)[N], const Args&... args) {
    (void)((void)args, ...);  // Suppress unused parameter warnings for all args
    return update_format_string<N, Args...>(format);
}

namespace detail {

// Convenience alias: the argument types as a tuple
template <typename... Args>
using arg_tuple = std::tuple<Args...>;

// Get size_in_bytes for argument at index I (0-based) in Args...
template <std::size_t I, typename... Args>
constexpr uint32_t size_of_arg() {
    // Use your existing get_types_info<Args...>()
    constexpr auto infos = get_types_info<Args...>();
    return infos[I].size_in_bytes;
}

// A list of indices as a type
template <std::size_t... Idxs>
struct index_list {
    static constexpr std::size_t size = sizeof...(Idxs);
};

// Insert an index at the front of an index_list
template <std::size_t NewHead, typename List>
struct push_front;

template <std::size_t NewHead, std::size_t... Idxs>
struct push_front<NewHead, index_list<Idxs...>> {
    using type = index_list<NewHead, Idxs...>;
};

// Insert index NewIdx into an already-sorted index_list, descending by size_in_bytes
template <std::size_t NewIdx, typename List, typename... Args>
struct insert_sorted;

// Base case: insert into empty list
template <std::size_t NewIdx, typename... Args>
struct insert_sorted<NewIdx, index_list<>, Args...> {
    using type = index_list<NewIdx>;
};

// Recursive case: compare with Head, then either insert before Head or recurse into Tail
template <std::size_t NewIdx, std::size_t Head, std::size_t... Tail, typename... Args>
struct insert_sorted<NewIdx, index_list<Head, Tail...>, Args...> {
    static constexpr uint32_t size_new = size_of_arg<NewIdx, Args...>();
    static constexpr uint32_t size_head = size_of_arg<Head, Args...>();

    // We want larger sizes first (descending)
    static constexpr bool before_head = (size_new > size_head);

    using type = std::conditional_t<
        before_head,
        // NewIdx is larger, place it before Head
        index_list<NewIdx, Head, Tail...>,
        // Otherwise keep Head, and insert into Tail...
        typename push_front<Head, typename insert_sorted<NewIdx, index_list<Tail...>, Args...>::type>::type>;
};

// Build sorted indices 0..N-1 using insertion sort
template <std::size_t N, std::size_t K, typename List, typename... Args>
struct build_sorted_indices_impl;

// Recursive step: insert K, then proceed with K+1
template <std::size_t N, std::size_t K, typename List, typename... Args>
struct build_sorted_indices_impl {
    using list_with_k = typename insert_sorted<K, List, Args...>::type;
    using type = typename build_sorted_indices_impl<N, K + 1, list_with_k, Args...>::type;
};

// Stop when K == N
template <std::size_t N, typename List, typename... Args>
struct build_sorted_indices_impl<N, N, List, Args...> {
    using type = List;
};

// Public entry: start with empty list and K = 0
template <std::size_t N, typename... Args>
struct build_sorted_indices {
    using type = typename build_sorted_indices_impl<N, 0, index_list<>, Args...>::type;
};

// Convert index_list<Idxs...> to std::index_sequence<Idxs...>
template <typename List>
struct index_list_to_seq;

template <std::size_t... Idxs>
struct index_list_to_seq<index_list<Idxs...>> {
    using type = std::index_sequence<Idxs...>;
};

}  // namespace detail

template <typename... Args>
struct arg_reorder_seq {
private:
    static constexpr std::size_t N = sizeof...(Args);
    using sorted_list = typename detail::build_sorted_indices<N, Args...>::type;

public:
    using type = typename detail::index_list_to_seq<sorted_list>::type;
};

template <typename... Args>
using arg_reorder_seq_t = typename arg_reorder_seq<Args...>::type;

}  // namespace formatting

namespace locking {

uint32_t wait_for_space(volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer, uint32_t message_size);
void release_lock();

// Takes lock unconditionally. Prints kernel id message if needed.
void acquire_lock() {
#if defined(ARCH_WORMHOLE)
    volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);

    while (true) {
    again:
        // Wait until lock is free (0)
        while (*lock_ptr != 0) {
            invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
            internal_::risc_context_switch();
#endif
        }

        // Write risc_id to lock to attempt to acquire it
        *lock_ptr = PROCESSOR_INDEX + 1;  // Use 1-based index to avoid writing 0 which is the free state

        // Make sure the write has propagated and other riscs see the updated value.
        invalidate_l1_cache();
        if (*lock_ptr != PROCESSOR_INDEX + 1) {
            goto again;
        }

        // One last check that we have successfully acquired the lock.
        invalidate_l1_cache();
        if (*lock_ptr == PROCESSOR_INDEX + 1) {
            break;  // Successfully acquired lock
        }
    }
#else
    auto& lock_atomic = get_device_print_buffer()->aux.lock;

    while (lock_atomic.exchange(1) != 0) {
        // Failed to acquire lock, wait and try again
        invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
    }
#endif

    // After acquiring the lock, invalidate our L1 cache to ensure we see the most up-to-date data in the buffer
    invalidate_l1_cache();

    // Check if we should print kernel id
    volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer = get_device_print_buffer();
    if (device_print_buffer->aux.wpos != DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        auto risc_state = device_print_buffer->aux.risc_state[PROCESSOR_INDEX];
        if (risc_state != DevicePrintRiscCoreState::PrintingDisabled) {
            if (risc_state == DevicePrintRiscCoreState::KernelNotPrinted) {
                uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
                tt_l1_ptr launch_msg_t* const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
                auto kernel_id = launch_msg->kernel_config.watcher_kernel_ids[PROCESSOR_INDEX];
                structures::DevicePrintHeader new_kernel_message = {};
                new_kernel_message.is_kernel = 1;
                new_kernel_message.risc_id = PROCESSOR_INDEX;
                new_kernel_message.message_payload = structures::DevicePrintHeader::max_message_payload_size;
                new_kernel_message.info_id = kernel_id;
                auto header_value = new_kernel_message.value;
                wait_for_space(device_print_buffer, sizeof(new_kernel_message));
                auto write_position = device_print_buffer->aux.wpos;
                auto device_print_buffer_ptr = &(device_print_buffer->data[0]) + write_position;
                formatting::device_print_type<decltype(header_value)>::serialize(
                    device_print_buffer_ptr, 0, header_value);
                device_print_buffer->aux.wpos += sizeof(new_kernel_message);
                device_print_buffer->aux.risc_state[PROCESSOR_INDEX] = DevicePrintRiscCoreState::KernelPrinted;
            }
        }
    }
}

void update_kernel_finished() {
    volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer = get_device_print_buffer();
    if (device_print_buffer->aux.risc_state[PROCESSOR_INDEX] != DevicePrintRiscCoreState::PrintingDisabled) {
        device_print_buffer->aux.risc_state[PROCESSOR_INDEX] = DevicePrintRiscCoreState::KernelNotPrinted;
    }
}

void release_lock() {
#if defined(ARCH_WORMHOLE)
    volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);

    asm volatile("" ::: "memory");
    *lock_ptr = 0;  // Release lock by setting to 0
    asm volatile("" ::: "memory");
#else
    auto& lock_atomic = get_device_print_buffer()->aux.lock;
    lock_atomic = 0;
#endif
}

void initialize_lock() {
#if defined(ARCH_WORMHOLE)
    volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);
    asm volatile("" ::: "memory");
    *lock_ptr = 0;  // Ensure lock starts in free state
    asm volatile("" ::: "memory");
#else
    auto& lock_atomic = get_device_print_buffer()->aux.lock;
    lock_atomic = 0;
#endif
}

uint32_t wait_for_space(volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer, uint32_t message_size) {
    // Read pointers
    auto write_position = device_print_buffer->aux.wpos;
    auto read_position = device_print_buffer->aux.rpos;

    // Check if it is starting magic
    if (write_position == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
        // Initialize valid state after print server starting magic.
        device_print_buffer->aux.wpos = 0;
        device_print_buffer->aux.rpos = 0;
        return 0;
    }

    if (write_position == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        // If we are in disabled state, return immediately without waiting for space.
        return 0;
    }

    // Check if there is enough space for the message until end of the buffer.
    if (write_position + message_size > sizeof(device_print_buffer->data)) {
        // It is important not to perform wrap around while reader position is at the beginning of the buffer,
        // as it will be the same state as at the beginning when buffer is empty. So in order to distinguish real
        // empty state and wrap around state, we will wait for reader to progress from start.
        if (read_position == 0) {
            // Reader is at the beginning, we need to wait for it to move before we can safely wrap around.
            WAYPOINT("DPW");
            while (read_position == 0) {
                invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
                internal_::risc_context_switch();
#endif
                // If we've closed the device, we've now disabled printing on it, don't hang.
                if (device_print_buffer->aux.wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                    return 0;
                };

                // Read new read position for next check
                read_position = device_print_buffer->aux.rpos;
            }
            WAYPOINT("DPD");
        }

        // Check if we should wair for reader to consume until end of the buffer before we can wrap around.
        if (write_position < read_position) {
            WAYPOINT("DPW");
            while (write_position < read_position) {
                invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
                internal_::risc_context_switch();
#endif
                // If we've closed the device, we've now disabled printing on it, don't hang.
                if (device_print_buffer->aux.wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                    return 0;
                };

                // Read new read position for next check
                read_position = device_print_buffer->aux.rpos;
            }
            WAYPOINT("DPD");
        }

        // There is not enough space for our message until end of buffer.
        // Check if we should add wrap around message in the buffer.
        if (write_position + sizeof(structures::DevicePrintHeader::value) <= sizeof(device_print_buffer->data)) {
            // We can fit a wrap around message, write it now so reader can process it while we wait for space.
            structures::DevicePrintHeader wrap_header = {};
            wrap_header.is_kernel = 0;
            wrap_header.risc_id = 0;
            wrap_header.message_payload = 0;
            wrap_header.info_id = structures::DevicePrintHeader::max_info_id_value;
            auto value = wrap_header.value;
            *reinterpret_cast<device_print_buffer_ptr<decltype(value)>>(device_print_buffer->data + write_position) =
                value;
        }
        // Wrap around to the beginning of the buffer and continue waiting for space there.
        write_position = device_print_buffer->aux.wpos = 0;
    } else if (write_position > read_position) {
        // There is enough space in the buffer.
        return write_position;
    }

    // Check if there is enough space between wpos and rpos
    if (write_position < read_position) {
        // Wrapped around, check if there is enough space between wpos and rpos
        WAYPOINT("DPW");
        while (write_position < read_position && write_position + message_size >= read_position) {
            invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
            internal_::risc_context_switch();
#endif
            // If we've closed the device, we've now disabled printing on it, don't hang.
            if (device_print_buffer->aux.wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                return 0;
            };

            // Read new read position for next check
            read_position = device_print_buffer->aux.rpos;
        }
        WAYPOINT("DPD");
    }
    return write_position;
}

}  // namespace locking

namespace serialization {

// Helper to serialize a single argument based on its type info
template <typename ArgumentType>
void serialize_argument(
    volatile tt_l1_ptr uint8_t* device_print_buffer, uint32_t offset, const ArgumentType& argument) {
    using base_type = std::remove_cv_t<std::remove_reference_t<ArgumentType>>;
    formatting::device_print_type<base_type>::serialize(device_print_buffer, offset, argument);
}

template <typename Tuple, typename ArgOffsetsType, std::size_t... Is>
inline void serialize_arguments_impl(
    volatile tt_l1_ptr uint8_t* device_print_buffer,
    Tuple&& tup,
    ArgOffsetsType arg_offsets,
    std::index_sequence<Is...>) {
    (serialize_argument(device_print_buffer, arg_offsets[Is], std::get<Is>(std::forward<Tuple>(tup))), ...);
}

template <typename... Args>
void serialize_arguments(volatile tt_l1_ptr uint8_t* device_print_buffer, Args&&... args) {
    auto tup = std::forward_as_tuple(std::forward<Args>(args)...);

    constexpr auto arg_offsets = formatting::get_arg_offsets<Args...>();
    using seq = formatting::arg_reorder_seq_t<Args...>;  // std::index_sequence<perm[0], perm[1], ...>

    serialize_arguments_impl(device_print_buffer, tup, arg_offsets, seq{});
}

template <typename... Args>
constexpr uint32_t get_total_message_size(Args&&...) {
    constexpr auto type_infos = formatting::get_types_info<Args...>();
    uint32_t total_size = sizeof(structures::DevicePrintHeader::value);  // Start with header size
    for (size_t i = 0; i < sizeof...(Args); ++i) {
        total_size += type_infos[i].size_in_bytes;
    }
    if (total_size % 4 != 0) {
        total_size += 4 - (total_size % 4);  // Pad to next multiple of 4 bytes
    }
    return total_size;
}

}  // namespace serialization

// Mark as noinline to ensure this function is not inlined, which causes smaller code to be generated (single JAL
// instruction for function call and two instructions for arguments).
__attribute__((noinline)) uint32_t
begin_message_write(structures::DevicePrintHeader header, std::uintptr_t string_info_address) {
    // Get buffer lock (once we change to be single buffer per L1 instead of per risc)
    locking::acquire_lock();

    // Check if we need to wrap buffer and wait for enough space in it
    volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer = get_device_print_buffer();
    uint32_t message_size = sizeof(header.value) + header.message_payload;
    auto write_position = locking::wait_for_space(device_print_buffer, message_size);

    // Update header
    std::uintptr_t string_info_start_address = reinterpret_cast<std::uintptr_t>(__device_print_strings_info_start);
    string_info_address -= string_info_start_address;
    std::uintptr_t string_info_index = string_info_address / sizeof(structures::DevicePrintStringInfo);
    using DevicePrintHeaderType = structures::DevicePrintHeader;
    if (string_info_index > DevicePrintHeaderType::max_info_id_value) {
        header.info_id = DevicePrintHeaderType::max_info_id_value;
    } else {
        header.info_id = static_cast<uint32_t>(string_info_index);
    }

    // Serialize header
    auto device_print_buffer_ptr = &(device_print_buffer->data[0]) + write_position;
    formatting::device_print_type<decltype(header.value)>::serialize(device_print_buffer_ptr, 0, header.value);

    return write_position;
}

// Mark as noinline to ensure this function is not inlined, which causes smaller code to be generated (single JAL
// instruction).
__attribute__((noinline)) void end_message_write() {
    // By this point, message is already serialized in the buffer. Read message header to get message size for moving
    // write pointer. We do this to minimize code size for calling end_message_write. We already know in the compile
    // time size of the message, but if we pass it as an argument to end_message_write, it will generate code to move
    // that argument (one more instruction). Here, since we don't care about code execution time, but code size, we read
    // the message header back from the buffer to get the message size, which allows us to avoid passing message size as
    // an argument and save some code size.
    volatile tt_l1_ptr DevicePrintMemoryLayout* device_print_buffer = get_device_print_buffer();
    auto write_position = device_print_buffer->aux.wpos;
    if (device_print_buffer->aux.wpos != DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        auto message_header_value =
            *reinterpret_cast<device_print_buffer_ptr<decltype(structures::DevicePrintHeader::value)>>(
                device_print_buffer->data + write_position);
        structures::DevicePrintHeader message_header;
        message_header.value = message_header_value;
        uint32_t message_size = sizeof(message_header.value) + message_header.message_payload;
        // Move write pointer in device_print buffer
        device_print_buffer->aux.wpos = write_position + message_size;
    }

    // Release buffer lock
    locking::release_lock();
}

}  // namespace device_print_detail

#else

#define DEVICE_PRINT(format, ...)
#define DEVICE_PRINT_INITIALIZE_LOCK()
#define DEVICE_PRINT_KERNEL_FINISHED()

#endif
