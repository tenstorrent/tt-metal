// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "new_dprint_structures.h"

#define NEW_DPRINT_STRINGS_SECTION_NAME "dprint_strings"
#define NEW_DPRINT_STRINGS_METADATA_SECTION_NAME "dprint_strings_metadata"

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

#define NEW_DPRINT(format, ...)                                                                                       \
    {                                                                                                                 \
        /* Validate format string syntax */                                                                           \
        static_assert(                                                                                                \
            dprint_detail::is_valid_format_string(format),                                                            \
            "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");                         \
        /* Validate placeholder format */                                                                             \
        static_assert(                                                                                                \
            !dprint_detail::has_mixed_placeholders(format),                                                           \
            "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");                  \
        /* For indexed placeholders, validate no index exceeds argument count */                                      \
        static_assert(                                                                                                \
            !dprint_detail::has_indexed_placeholders(format) ||                                                       \
                dprint_detail::get_max_index(format) < static_cast<int>(dprint_detail::count_arguments(__VA_ARGS__)), \
            "Placeholder index exceeds number of arguments");                                                         \
        /* For indexed placeholders, validate all arguments are referenced */                                         \
        static_assert(                                                                                                \
            !dprint_detail::has_indexed_placeholders(format) ||                                                       \
                dprint_detail::all_arguments_referenced(format, dprint_detail::count_arguments(__VA_ARGS__)),         \
            "All arguments must be referenced when using indexed placeholders");                                      \
        /* TODO: Validate correctness of format and arguments */                                                      \
        /* TODO: Update format to include all necessary data and store it into dprint section */                      \
        /* TODO: Write dprint message to dprint buffer */                                                             \
    }

namespace dprint_detail {

template <typename... Args>
constexpr std::size_t count_arguments(const Args&...) {
    return sizeof...(Args);
}

// Helper to check if a character is a digit
constexpr bool is_digit(char c) { return c >= '0' && c <= '9'; }

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
    while (pos < format_len && is_digit(format[pos])) {
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

        if (is_digit(format[i + 1])) {
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

// Helper to validate format string for invalid brace sequences
// Returns true if format string is valid, false if it contains errors
template <std::size_t N>
constexpr bool is_valid_format_string(const char (&format)[N]) {
    for (std::size_t i = 0; i < N - 1;) {
        FormatToken token = parse_format_token(format, i);

        // Check for invalid placeholder syntax
        if (token.type == TokenType::InvalidPlaceholder) {
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
        FormatToken token = parse_format_token(format, i);
        if (token.type == TokenType::Placeholder) {
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
        FormatToken token = parse_format_token(format, i);
        if (token.type == TokenType::Placeholder && token.is_indexed()) {
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
        FormatToken token = parse_format_token(format, i);
        if (token.type == TokenType::Placeholder && token.is_indexed()) {
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

    // Track which arguments are referenced (up to 100 arguments)
    bool referenced[100] = {};
    if (arg_count > 100) {
        return false;  // Limit for simplicity
    }

    for (std::size_t i = 0; i < N - 1;) {
        FormatToken token = parse_format_token(format, i);
        if (token.type == TokenType::Placeholder && token.is_indexed()) {
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

}  // namespace dprint_detail
