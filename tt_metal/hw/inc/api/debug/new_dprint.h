// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "new_dprint_structures.h"
#include "hostdevcommon/new_dprint_common.h"
#include "stream_io_map.h"

#define NEW_DPRINT_STRINGS_SECTION_NAME ".dprint_strings"
#define NEW_DPRINT_STRINGS_INFO_SECTION_NAME ".dprint_strings_info"

#ifdef UCK_CHLKC_UNPACK
#define NEW_DPRINT_UNPACK(format, ...) NEW_DPRINT(format, __VA_ARGS__)
#else
#define NEW_DPRINT_UNPACK(format, ...)
#endif

#ifdef UCK_CHLKC_MATH
#define NEW_DPRINT_MATH(format, ...) NEW_DPRINT(format, __VA_ARGS__)
#else
#define NEW_DPRINT_MATH(format, ...)
#endif

#ifdef UCK_CHLKC_PACK
#define NEW_DPRINT_PACK(format, ...) NEW_DPRINT(format, __VA_ARGS__)
#else
#define NEW_DPRINT_PACK(format, ...)
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#define NEW_DPRINT_DATA0(format, ...)    \
    if (noc_index == 0) {                \
        NEW_DPRINT(format, __VA_ARGS__); \
    }
#define NEW_DPRINT_DATA1(format, ...)    \
    if (noc_index == 1) {                \
        NEW_DPRINT(format, __VA_ARGS__); \
    }
#else
#define NEW_DPRINT_DATA0(format, ...)
#define NEW_DPRINT_DATA1(format, ...)
#endif

#if defined(KERNEL_BUILD)
#define NEW_DPRINT_IS_KERNEL 1
#else
#define NEW_DPRINT_IS_KERNEL 0
#endif

#define NEW_DPRINT_GET_STRING_INDEX(variable_name, updated_format)                                           \
    {                                                                                                        \
        static const auto allocated_string __attribute__((section(NEW_DPRINT_STRINGS_SECTION_NAME), used)) = \
            updated_format.to_array();                                                                       \
        static const auto allocated_file_string                                                              \
            __attribute__((section(NEW_DPRINT_STRINGS_SECTION_NAME), used)) = []() {                         \
                dprint_detail::helpers::static_string<sizeof(__FILE__)> file_str;                            \
                for (std::size_t i = 0; i < sizeof(__FILE__); ++i) {                                         \
                    file_str.push_back(__FILE__[i]);                                                         \
                }                                                                                            \
                return file_str.to_array();                                                                  \
            }();                                                                                             \
        static dprint_detail::structures::DPrintStringInfo allocated_string_info                             \
            __attribute__((section(NEW_DPRINT_STRINGS_INFO_SECTION_NAME), used)) = {                         \
                allocated_string.data(), allocated_file_string.data(), __LINE__};                            \
    }                                                                                                        \
    constexpr uint32_t variable_name = __COUNTER__;

#define NEW_DPRINT(format, ...)                                                                                        \
    {                                                                                                                  \
        /* Validate format string syntax */                                                                            \
        static_assert(                                                                                                 \
            dprint_detail::checks::is_valid_format_string(format),                                                     \
            "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");                          \
        /* Validate placeholder format */                                                                              \
        static_assert(                                                                                                 \
            !dprint_detail::checks::has_mixed_placeholders(format),                                                    \
            "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");                   \
        /* For indexed placeholders, validate no index exceeds argument count */                                       \
        static_assert(                                                                                                 \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                                                \
                dprint_detail::checks::get_max_index(format) < dprint_detail::helpers::count_arguments(__VA_ARGS__),   \
            "Placeholder index exceeds number of arguments");                                                          \
        /* For indexed placeholders, validate all arguments are referenced */                                          \
        static_assert(                                                                                                 \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                                                \
                dprint_detail::checks::all_arguments_referenced(format, __VA_ARGS__),                                  \
            "All arguments must be referenced when using indexed placeholders");                                       \
        /* For non-indexed placeholders, count must match argument count */                                            \
        static_assert(                                                                                                 \
            dprint_detail::checks::has_indexed_placeholders(format) ||                                                 \
                dprint_detail::checks::count_placeholders(format) ==                                                   \
                    dprint_detail::helpers::count_arguments(__VA_ARGS__),                                              \
            "Number of {} placeholders must match number of arguments");                                               \
        /* TODO: In case we decide to optimize write into dprint buffer, we might want to reorder arguments. This */   \
        /* will influence serialization and format update. Server side will remain the same.*/                         \
        /* Update format to include all necessary data */                                                              \
        constexpr auto updated_format =                                                                                \
            dprint_detail::formatting::update_format_string_from_args(format, __VA_ARGS__);                            \
        /* Store updated format string in a special section for dprint */                                              \
        NEW_DPRINT_GET_STRING_INDEX(dprint_info_index, updated_format);                                                \
        /* Get buffer lock (once we change to be single buffer per L1 instead of per risc)*/                           \
        dprint_detail::locking::acquire_lock();                                                                        \
        /* Get dprint buffer*/                                                                                         \
        volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer = get_new_debug_print_buffer();                       \
        /* Check if we need to wrap buffer and wait for enough space in it */                                          \
        constexpr auto message_size = dprint_detail::serialization::get_total_message_size(__VA_ARGS__);               \
        dprint_detail::locking::wait_for_space(dprint_buffer, message_size);                                           \
        /* Generate dprint message header */                                                                           \
        dprint_detail::structures::DPrintHeader header = {};                                                           \
        header.is_kernel = NEW_DPRINT_IS_KERNEL;                                                                       \
        header.risc_id = PROCESSOR_INDEX;                                                                              \
        header.message_payload = message_size - sizeof(header); /* Payload size does not include header itself */      \
        static_assert(                                                                                                 \
            dprint_info_index <= dprint_detail::structures::DPrintHeader::max_info_id_value,                           \
            "Too many DPRINT calls, exceeds limit");                                                                   \
        header.info_id = dprint_info_index;                                                                            \
        auto header_value = header.value;                                                                              \
        /* Serialize message */                                                                                        \
        auto dprint_buffer_ptr = &(dprint_buffer->data[0]) + dprint_buffer->aux.wpos;                                  \
        dprint_detail::formatting::dprint_type<decltype(header_value)>::serialize(dprint_buffer_ptr, 0, header_value); \
        dprint_detail::serialization::serialize_arguments(dprint_buffer_ptr, __VA_ARGS__);                             \
        /* Move write pointer in dprint buffer */                                                                      \
        dprint_buffer->aux.wpos += message_size;                                                                       \
        /* Release buffer lock */                                                                                      \
        dprint_detail::locking::release_lock();                                                                        \
    }

namespace dprint_detail {

template <typename BufferType>
using dprint_buffer_ptr = volatile tt_l1_ptr BufferType*;

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

struct dprint_type_info {
    char type_char;
    uint32_t size_in_bytes;
};

// Type-to-info mapping for format strings and serialization
template <typename T>
struct dprint_type {
    static constexpr dprint_type_info value = {'#', 0};  // Unknown type default
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, T argument) {
        static_assert(true, "No serialization defined for this type");
    }
};

// Specializations for different types
template <>
struct dprint_type<std::int8_t> {
    static constexpr dprint_type_info value = {'b', sizeof(std::int8_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, int8_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<int8_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::uint8_t> {
    static constexpr dprint_type_info value = {'B', sizeof(std::uint8_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, uint8_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<uint8_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::int16_t> {
    static constexpr dprint_type_info value = {'h', sizeof(std::int16_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, int16_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<int16_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::uint16_t> {
    static constexpr dprint_type_info value = {'H', sizeof(std::uint16_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, uint16_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<uint16_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::int32_t> {
    static constexpr dprint_type_info value = {'i', sizeof(std::int32_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, int32_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<int32_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::uint32_t> {
    static constexpr dprint_type_info value = {'I', sizeof(std::uint32_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, uint32_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<uint32_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::int64_t> {
    static constexpr dprint_type_info value = {'q', sizeof(std::int64_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, int64_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<int64_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<std::uint64_t> {
    static constexpr dprint_type_info value = {'Q', sizeof(std::uint64_t)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, uint64_t argument) {
        *reinterpret_cast<dprint_buffer_ptr<uint64_t>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<float> {
    static constexpr dprint_type_info value = {'f', sizeof(float)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, float argument) {
        *reinterpret_cast<dprint_buffer_ptr<float>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<double> {
    static constexpr dprint_type_info value = {'d', sizeof(double)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, double argument) {
        *reinterpret_cast<dprint_buffer_ptr<double>>(dprint_buffer + offset) = argument;
    }
};
template <>
struct dprint_type<bool> {
    static constexpr dprint_type_info value = {'?', 1};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, bool argument) {
        *(dprint_buffer + offset) = static_cast<uint8_t>(argument);
    }
};

// Pointer types (including strings)
template <typename T>
struct dprint_type<T*> {
    static constexpr dprint_type_info value = {'p', sizeof(T*)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, T* argument) {
        if constexpr (sizeof(T*) == 4) {
            *reinterpret_cast<dprint_buffer_ptr<uint32_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(T*) == 8) {
            *reinterpret_cast<dprint_buffer_ptr<uint64_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(T*) == 4 || sizeof(T*) == 8, "Unsupported pointer size");
        }
    }
};
template <>
struct dprint_type<char*> {
    static constexpr dprint_type_info value = {'s', sizeof(char*)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, char* argument) {
        if constexpr (sizeof(char*) == 4) {
            *reinterpret_cast<dprint_buffer_ptr<uint32_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(char*) == 8) {
            *reinterpret_cast<dprint_buffer_ptr<uint64_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(char*) == 4 || sizeof(char*) == 8, "Unsupported pointer size");
        }
    }
};
template <>
struct dprint_type<const char*> {
    static constexpr dprint_type_info value = {'s', sizeof(const char*)};
    static void serialize(dprint_buffer_ptr<uint8_t> dprint_buffer, uint32_t offset, const char* argument) {
        if constexpr (sizeof(const char*) == 4) {
            *reinterpret_cast<dprint_buffer_ptr<uint32_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint32_t>(argument);
        } else if constexpr (sizeof(const char*) == 8) {
            *reinterpret_cast<dprint_buffer_ptr<uint64_t>>(dprint_buffer + offset) =
                reinterpret_cast<uint64_t>(argument);
        } else {
            static_assert(sizeof(const char*) == 4 || sizeof(const char*) == 8, "Unsupported pointer size");
        }
    }
};

// Array types (treat as strings)
template <std::size_t N>
struct dprint_type<char[N]> {
    static constexpr dprint_type_info value = {'s', sizeof(const char*)};
};
template <std::size_t N>
struct dprint_type<const char[N]> {
    static constexpr dprint_type_info value = {'s', sizeof(const char*)};
};

// Helper to get type character for a single type, removing cv-qualifiers and references
template <typename T>
constexpr dprint_type_info get_type_info() {
    using base_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return dprint_type<base_type>::value;
}

template <typename... Args>
constexpr std::array<dprint_type_info, sizeof...(Args)> get_types_info() {
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
    std::array<uint32_t, sizeof...(Args)> arg_offset = {};
    uint32_t current_offset = sizeof(dprint_detail::structures::DPrintHeader::value);
    for (std::size_t i = 0; i < arg_offset.size(); ++i) {
        arg_offset[i] = current_offset;
        current_offset += type_infos[arg_reorder[i]].size_in_bytes;
    }
    return arg_offset;
}

// Main function to update format string with type information
// Supports both {} and {N} placeholder styles (fmtlib-compatible)
template <std::size_t N, typename... Args>
constexpr auto update_format_string(const char (&format)[N]) {
    constexpr std::size_t format_len = N - 1;  // Exclude null terminator

    // Calculate maximum result length:
    // - Original format length
    // - Each {} or {N} can add at most 2 extra characters (":X")
    // - Assuming worst case of format_len/2 placeholders (every other char is {)
    // Use a reasonable upper bound
    constexpr std::size_t result_len = format_len + (format_len / 2 + 1) * 2;

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

// TODO: IMPORTANT!!! We need to initialize dprint sync register to 0 during startup (probably on brisc).

void acquire_lock() {
    volatile uint32_t* lock_ptr = get_dprint_sync_register_ptr();

    while (true) {
        // Wait until lock is free (0)
        while (*lock_ptr != 0) {
#if defined(COMPILE_FOR_ERISC)
            internal_::risc_context_switch();
#endif
        }

        // Write risc_id to lock to attempt to acquire it
        *lock_ptr = PROCESSOR_INDEX + 1;  // Use 1-based index to avoid writing 0 which is the free state

        // TODO: Figure out how many queries we need here to ensure the write has propagated and other riscs see the
        // updated value.
        if (*lock_ptr != PROCESSOR_INDEX + 1) {
            continue;
        }
        if (*lock_ptr != PROCESSOR_INDEX + 1) {
            continue;
        }
        if (*lock_ptr != PROCESSOR_INDEX + 1) {
            continue;
        }

        // If after several checks the lock value is still what we set, we have successfully acquired the lock.
        if (*lock_ptr == PROCESSOR_INDEX + 1) {
            break;  // Successfully acquired lock
        }
    }
}

void release_lock() {
    volatile uint32_t* lock_ptr = get_dprint_sync_register_ptr();

    asm volatile("" ::: "memory");
    *lock_ptr = 0;  // Release lock by setting to 0
    asm volatile("" ::: "memory");
}

void initialize_lock() {
    volatile uint32_t* lock_ptr = get_dprint_sync_register_ptr();
    asm volatile("" ::: "memory");
    *lock_ptr = 0;  // Ensure lock starts in free state
    asm volatile("" ::: "memory");
}

void wait_for_space(volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer, uint32_t message_size) {
    // Check if we are wrapped around
    if (dprint_buffer->aux.wpos > dprint_buffer->aux.rpos) {
        // We are writing in front of the read position. Check if we need to wrap around.
        if (dprint_buffer->aux.wpos + message_size >= sizeof(dprint_buffer->data)) {
            // There is not enough space for our message until end of buffer.
            // Check if we should add wrap around message in the buffer.
            if (dprint_buffer->aux.wpos <
                sizeof(dprint_buffer->data) - sizeof(dprint_detail::structures::DPrintHeader::value)) {
                // We can fit a wrap around message, write it now so reader can process it while we wait for space.
                dprint_detail::structures::DPrintHeader wrap_header = {};
                wrap_header.is_kernel = 0;
                wrap_header.risc_id = 0;
                wrap_header.message_payload = 0;
                wrap_header.info_id = dprint_detail::structures::DPrintHeader::max_info_id_value;
                auto value = wrap_header.value;
                *reinterpret_cast<dprint_buffer_ptr<decltype(value)>>(dprint_buffer->data + dprint_buffer->aux.wpos) =
                    value;
            }
            // Wrap around to the beginning of the buffer and continue waiting for space there.
            dprint_buffer->aux.wpos = 0;
        } else {
            // There is enough space in the buffer.
            return;
        }
    }

    // Check if there is enough space between wpos and rpos
    if (dprint_buffer->aux.wpos < dprint_buffer->aux.rpos) {
        // Wrapped around, check if there is enough space between wpos and rpos
        WAYPOINT("DPW");
        while (dprint_buffer->aux.wpos < dprint_buffer->aux.rpos &&
               dprint_buffer->aux.rpos < dprint_buffer->aux.wpos + message_size) {
            invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
            internal_::risc_context_switch();
#endif
            // If we've closed the device, we've now disabled printing on it, don't hang.
            if (dprint_buffer->aux.wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                return;
            };  // wait for host to catch up to wpos with it's rpos
        }
        WAYPOINT("DPD");
    }
}

}  // namespace locking

namespace serialization {

// Helper to serialize a single argument based on its type info
template <typename ArgumentType>
void serialize_argument(volatile tt_l1_ptr uint8_t* dprint_buffer, uint32_t offset, const ArgumentType& argument) {
    using base_type = std::remove_cv_t<std::remove_reference_t<ArgumentType>>;
    formatting::dprint_type<base_type>::serialize(dprint_buffer, offset, argument);
}

template <typename Tuple, typename ArgOffsetsType, std::size_t... Is>
inline void serialize_arguments_impl(
    volatile tt_l1_ptr uint8_t* dprint_buffer, Tuple&& tup, ArgOffsetsType arg_offsets, std::index_sequence<Is...>) {
    (serialize_argument(dprint_buffer, arg_offsets[Is], std::get<Is>(std::forward<Tuple>(tup))), ...);
}

template <typename... Args>
void serialize_arguments(volatile tt_l1_ptr uint8_t* dprint_buffer, Args&&... args) {
    auto tup = std::forward_as_tuple(std::forward<Args>(args)...);

    constexpr auto arg_offsets = formatting::get_arg_offsets<Args...>();
    using seq = formatting::arg_reorder_seq_t<Args...>;  // std::index_sequence<perm[0], perm[1], ...>

    serialize_arguments_impl(dprint_buffer, tup, arg_offsets, seq{});
}

template <typename... Args>
constexpr uint32_t get_total_message_size(Args&&...) {
    constexpr auto type_infos = formatting::get_types_info<Args...>();
    uint32_t total_size = sizeof(dprint_detail::structures::DPrintHeader::value);  // Start with header size
    for (size_t i = 0; i < sizeof...(Args); ++i) {
        total_size += type_infos[i].size_in_bytes;
    }
    if (total_size % 4 != 0) {
        total_size += 4 - (total_size % 4);  // Pad to next multiple of 4 bytes
    }
    return total_size;
}

}  // namespace serialization

}  // namespace dprint_detail
