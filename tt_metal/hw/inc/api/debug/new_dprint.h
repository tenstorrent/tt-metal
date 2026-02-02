// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "new_dprint_structures.h"

#define NEW_DPRINT_STRINGS_SECTION_NAME ".dprint_strings"
#define NEW_DPRINT_STRINGS_INFO_SECTION_NAME ".dprint_strings_info"
#define NEW_DPRINT_MAX_ARGUMENTS 100

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

#define NEW_DPRINT(format, ...)                                                                      \
    {                                                                                                \
        /* Validate format string syntax */                                                          \
        static_assert(                                                                               \
            dprint_detail::checks::is_valid_format_string(format),                                   \
            "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");        \
        /* Validate placeholder format */                                                            \
        static_assert(                                                                               \
            !dprint_detail::checks::has_mixed_placeholders(format),                                  \
            "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string"); \
        /* For indexed placeholders, validate no index exceeds argument count */                     \
        static_assert(                                                                               \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                              \
                dprint_detail::checks::get_max_index(format) <                                       \
                    static_cast<int>(dprint_detail::helpers::count_arguments(__VA_ARGS__)),          \
            "Placeholder index exceeds number of arguments");                                        \
        /* For indexed placeholders, validate all arguments are referenced */                        \
        static_assert(                                                                               \
            !dprint_detail::checks::has_indexed_placeholders(format) ||                              \
                dprint_detail::checks::all_arguments_referenced(                                     \
                    format, dprint_detail::helpers::count_arguments(__VA_ARGS__)),                   \
            "All arguments must be referenced when using indexed placeholders");                     \
        /* For non-indexed placeholders, count must match argument count */                          \
        static_assert(                                                                               \
            dprint_detail::checks::has_indexed_placeholders(format) ||                               \
                dprint_detail::checks::count_placeholders(format) ==                                 \
                    dprint_detail::helpers::count_arguments(__VA_ARGS__),                            \
            "Number of {} placeholders must match number of arguments");                             \
        /* Update format to include all necessary data */                                            \
        constexpr auto updated_format =                                                              \
            dprint_detail::formatting::update_format_string_from_args(format, ##__VA_ARGS__);        \
        /* Store updated format string in a special section for dprint */                            \
        NEW_DPRINT_GET_STRING_INDEX(dprint_info_index, updated_format);                              \
        static_assert(dprint_info_index <= 1024, "Too many DPRINT calls, exceeds limit");            \
        /* TODO: Write dprint message to dprint buffer */                                            \
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
    int value;
    std::size_t new_pos;
};

// Helper to parse an integer from format string starting at position i
// Returns the parsed value and the new position after the digits
constexpr IndexParseResult parse_index(const char* format, std::size_t i, std::size_t format_len) {
    int value = 0;
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
    TokenType type;
    std::size_t end_pos;  // Position to continue parsing from (after this token)

    // Placeholder-specific data
    int index;  // The N in {N}, or -1 for {} or escape sequences

    constexpr bool is_indexed() const { return index != -1; }
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
        return FormatToken{TokenType::RegularChar, i + 1, -1};
    }

    char c = format[i];

    // Check for escaped opening brace {{
    if (c == '{' && i + 1 < format_len && format[i + 1] == '{') {
        return FormatToken{TokenType::EscapedOpenBrace, i + 2, -1};
    }

    // Check for escaped closing brace }}
    if (c == '}' && i + 1 < format_len && format[i + 1] == '}') {
        return FormatToken{TokenType::EscapedCloseBrace, i + 2, -1};
    }

    // Check for placeholder {N} or {}
    if (c == '{') {
        if (i + 1 >= format_len) {
            // '{' at end of string is invalid
            return FormatToken{TokenType::InvalidPlaceholder, i + 1, -1};
        }

        if (helpers::is_digit(format[i + 1])) {
            // Indexed placeholder {N}
            IndexParseResult result = parse_index(format, i + 1, format_len);
            if (result.new_pos < format_len && format[result.new_pos] == '}') {
                return FormatToken{TokenType::Placeholder, result.new_pos + 1, result.value};
            }
            // Invalid: digit not followed by }
            return FormatToken{TokenType::InvalidPlaceholder, i + 1, -1};
        }
        if (format[i + 1] == '}') {
            // Non-indexed placeholder {}
            return FormatToken{TokenType::Placeholder, i + 2, -1};
        }

        // Invalid: '{' not followed by '{', '}', or digit
        return FormatToken{TokenType::InvalidPlaceholder, i + 1, -1};
    }

    // Regular character
    return FormatToken{TokenType::RegularChar, i + 1, -1};
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
constexpr int get_max_index(const char (&format)[N]) {
    int max_index = -1;
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            max_index = std::max(max_index, token.index);
        }
        i = token.end_pos;
    }
    return max_index;
}

// Helper to validate that all arguments are referenced in indexed format
// Returns true if all argument indices from 0 to arg_count-1 are used at least once
template <std::size_t N>
constexpr bool all_arguments_referenced(const char (&format)[N], std::size_t arg_count) {
    if (arg_count == 0) {
        return true;
    }

    // Track which arguments are referenced (up to NEW_DPRINT_MAX_ARGUMENTS arguments)
    bool referenced[NEW_DPRINT_MAX_ARGUMENTS] = {};
    if (arg_count > NEW_DPRINT_MAX_ARGUMENTS) {
        return false;  // Limit for simplicity
    }

    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            if (token.index >= 0 && static_cast<std::size_t>(token.index) < arg_count) {
                referenced[token.index] = true;
            }
        }
        i = token.end_pos;
    }

    // Check that all arguments from 0 to arg_count-1 are referenced
    for (std::size_t i = 0; i < arg_count; ++i) {
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
            int arg_index = token.is_indexed() ? token.index : type_index++;

            // Unified handling for both indexed and non-indexed: output {index:type}
            result.push_back('{');

            // Output the index
            static_assert(NEW_DPRINT_MAX_ARGUMENTS <= 100, "Adjust index output code for larger max arguments");
            if (arg_index >= 10) {
                result.push_back('0' + (arg_index / 10));
            }
            result.push_back('0' + (arg_index % 10));

            // Add colon and type character
            result.push_back(':');
            result.push_back(type_infos[arg_index].type_char);
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

}  // namespace dprint_detail
