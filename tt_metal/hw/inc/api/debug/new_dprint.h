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

#define NEW_DPRINT(format, ...)                                                                                      \
    {                                                                                                                \
        /* Validate format string syntax */                                                                          \
        static_assert(                                                                                               \
            dprint_detail::checks::is_valid_format_string(format),                                                   \
            "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");                        \
        /* Validate placeholder format */                                                                            \
        static_assert(                                                                                               \
            !dprint_detail::checks::has_mixed_placeholders(format),                                                  \
            "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");                 \
        /* For indexed placeholders, validate no index exceeds argument count */                                     \
        static_assert(                                                                                               \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                                              \
                dprint_detail::checks::get_max_index(format) <                                                       \
                    static_cast<int>(dprint_detail::helpers::count_arguments(__VA_ARGS__)),                          \
            "Placeholder index exceeds number of arguments");                                                        \
        /* For indexed placeholders, validate all arguments are referenced */                                        \
        static_assert(                                                                                               \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                                              \
                dprint_detail::checks::all_arguments_referenced(format, __VA_ARGS__),                                \
            "All arguments must be referenced when using indexed placeholders");                                     \
        /* For non-indexed placeholders, count must match argument count */                                          \
        static_assert(                                                                                               \
            dprint_detail::checks::has_indexed_placeholders(format) ||                                               \
                dprint_detail::checks::count_placeholders(format) ==                                                 \
                    dprint_detail::helpers::count_arguments(__VA_ARGS__),                                            \
            "Number of {} placeholders must match number of arguments");                                             \
        /* TODO: In case we decide to optimize write into dprint buffer, we might want to reorder arguments. This */ \
        /* will influence serialization and format update. Server side will remain the same.*/                       \
        /* Update format to include all necessary data */                                                            \
        constexpr auto updated_format =                                                                              \
            dprint_detail::formatting::update_format_string_from_args(format, ##__VA_ARGS__);                        \
        /* Store updated format string in a special section for dprint */                                            \
        NEW_DPRINT_GET_STRING_INDEX(dprint_info_index, updated_format);                                              \
        /* Get buffer lock (once we change to be single buffer per L1 instead of per risc)*/                         \
        dprint_detail::locking::acquire_lock();                                                                      \
        /* Generate dprint message header */                                                                         \
        dprint_detail::structures::DPrintHeader header = {};                                                         \
        header.is_kernel = NEW_DPRINT_IS_KERNEL;                                                                     \
        header.risc_id = PROCESSOR_INDEX;                                                                            \
        static_assert(                                                                                               \
            dprint_info_index <= dprint_detail::structures::DPrintHeader::max_info_id_value,                         \
            "Too many DPRINT calls, exceeds limit");                                                                 \
        header.info_id = dprint_info_index;                                                                          \
        uint16_t header_value = header.value;                                                                        \
        /* Get dprint buffer and write header and arguments */                                                       \
        volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer = get_new_debug_print_buffer();                     \
        dprint_detail::serialization::serialize_argument(dprint_buffer, header_value);                               \
        dprint_detail::serialization::serialize_arguments(dprint_buffer, ##__VA_ARGS__);                             \
        /* Release buffer lock */                                                                                    \
        dprint_detail::locking::release_lock();                                                                      \
    }

namespace dprint_detail {

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

template <typename T>
constexpr bool init_to_false() {
    return false;
}

// Helper to validate that all arguments are referenced in indexed format
// Returns true if all argument indices from 0 to arg_count-1 are used at least once
template <std::size_t N, typename... Args>
constexpr bool all_arguments_referenced(const char (&format)[N], const Args&... args) {
    if (sizeof...(args) == 0) {
        return true;
    }

    // Track which arguments are referenced
    bool referenced[] = {init_to_false<Args>()...};

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
};

// Specializations for different types
template <>
struct dprint_type<std::int8_t> {
    static constexpr dprint_type_info value = {'b', sizeof(std::int8_t)};
};
template <>
struct dprint_type<std::uint8_t> {
    static constexpr dprint_type_info value = {'B', sizeof(std::uint8_t)};
};
template <>
struct dprint_type<std::int16_t> {
    static constexpr dprint_type_info value = {'h', sizeof(std::int16_t)};
};
template <>
struct dprint_type<std::uint16_t> {
    static constexpr dprint_type_info value = {'H', sizeof(std::uint16_t)};
};
template <>
struct dprint_type<std::int32_t> {
    static constexpr dprint_type_info value = {'i', sizeof(std::int32_t)};
};
template <>
struct dprint_type<std::uint32_t> {
    static constexpr dprint_type_info value = {'I', sizeof(std::uint32_t)};
};
template <>
struct dprint_type<std::int64_t> {
    static constexpr dprint_type_info value = {'q', sizeof(std::int64_t)};
};
template <>
struct dprint_type<std::uint64_t> {
    static constexpr dprint_type_info value = {'Q', sizeof(std::uint64_t)};
};
template <>
struct dprint_type<float> {
    static constexpr dprint_type_info value = {'f', sizeof(float)};
};
template <>
struct dprint_type<double> {
    static constexpr dprint_type_info value = {'d', sizeof(double)};
};
template <>
struct dprint_type<bool> {
    static constexpr dprint_type_info value = {'?', 1};
};

// Pointer types (including strings)
template <typename T>
struct dprint_type<T*> {
    static constexpr dprint_type_info value = {'p', sizeof(T*)};
};
template <>
struct dprint_type<char*> {
    static constexpr dprint_type_info value = {'s', sizeof(char*)};
};
template <>
struct dprint_type<const char*> {
    static constexpr dprint_type_info value = {'s', sizeof(const char*)};
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

    constexpr dprint_type_info type_infos[] = {get_type_info<Args>()...};
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

            // Output the index
            result.push_back_uint32(arg_index);

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

    *lock_ptr = 0;  // Release lock by setting to 0
}

void initialize_lock() {
    volatile uint32_t* lock_ptr = get_dprint_sync_register_ptr();
    *lock_ptr = 0;  // Ensure lock starts in free state
}

}  // namespace locking

namespace serialization {

void serialize_argument(
    volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer, const uint8_t* argument_data, uint32_t size) {
    volatile uint8_t* start_pointer = dprint_buffer->data;
    volatile uint8_t* end_pointer = dprint_buffer->data + sizeof(dprint_buffer->data);
    volatile uint8_t* write_pointer = dprint_buffer->data + dprint_buffer->aux.wpos;
    volatile uint8_t* read_pointer = dprint_buffer->data + dprint_buffer->aux.rpos;

    if (write_pointer >= read_pointer) {
        if (write_pointer + size <= end_pointer) {
            // We have enough space to write the argument
            for (uint32_t i = 0; i < size; ++i) {
                write_pointer[i] = argument_data[i];
            }
            write_pointer += size;
            dprint_buffer->aux.wpos = write_pointer - start_pointer;
            return;
        }

        // Write what we can and then wrap around
        uint32_t space_to_end = end_pointer - write_pointer;
        if (space_to_end > 0) {
            for (uint32_t i = 0; i < space_to_end; ++i) {
                write_pointer[i] = argument_data[i];
            }
            argument_data += space_to_end;
            size -= space_to_end;
        }
        write_pointer = start_pointer;  // Wrap around to the beginning of the buffer
    }

    // Now write the remaining argument data at the beginning of the buffer
    if (write_pointer + size > read_pointer) {
        // We need to wait for the reader to advance, so that we don't overwrite unread data.
        WAYPOINT("DPW");
        while (dprint_buffer->aux.wpos + size > dprint_buffer->aux.rpos) {
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

    for (uint32_t i = 0; i < size; ++i) {
        write_pointer[i] = argument_data[i];
    }
    write_pointer += size;
    dprint_buffer->aux.wpos = write_pointer - start_pointer;
}

// Helper to serialize a single argument based on its type info
template <typename ArgumentType>
void serialize_argument(volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer, const ArgumentType& argument) {
    serialize_argument(
        dprint_buffer,
        reinterpret_cast<const uint8_t*>(&argument),
        formatting::get_type_info<ArgumentType>().size_in_bytes);
}

template <>
void serialize_argument(volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer, const bool& argument) {
    serialize_argument(dprint_buffer, static_cast<uint8_t>(argument ? 1 : 0));
}

// Variadic template to serialize all arguments in order
template <typename... Args>
void serialize_arguments(volatile tt_l1_ptr NewDebugPrintMemLayout* dprint_buffer, Args&&... args) {
    (serialize_argument(dprint_buffer, std::forward<Args>(args)), ...);
}

}  // namespace serialization

}  // namespace dprint_detail
