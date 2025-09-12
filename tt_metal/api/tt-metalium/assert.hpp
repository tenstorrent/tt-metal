// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <tt-logger/tt-logger.hpp>

namespace tt::assert {

/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
std::vector<std::string> backtrace(int size = 64, int skip = 1);

/**
 * @brief String to get current stack information
 * @param[in] size Maximum number of stacks
 * @param[in] skip Skip the number of layers at the top of the stack
 * @param[in] prefix Output before stack information
 */
std::string backtrace_to_string(int size = 64, int skip = 2, const std::string& prefix = "");

namespace detail{

template <typename... Args>
[[noreturn]] void tt_throw_impl(
    char const* file, int line, char const* assert_type, char const* condition_str, Args const&... args) {
    if (std::getenv("TT_ASSERT_ABORT")) {
        if constexpr (sizeof...(args) > 0) {
            log_critical(tt::LogAlways, args...);
        }
        abort();
    }

    std::stringstream trace_message_ss = {};
    trace_message_ss << assert_type << " @ " << file << ":" << line << ": " << condition_str << std::endl;
    if constexpr (sizeof...(args) > 0) {
        trace_message_ss << "info:" << std::endl;
        trace_message_ss << fmt::format(args...) << std::endl;
        log_critical(tt::LogAlways, args...);
    }
    trace_message_ss << "backtrace:\n";
    trace_message_ss << tt::assert::backtrace_to_string(100, 3, " --- ");
    trace_message_ss << std::flush;
    throw std::runtime_error(trace_message_ss.str());
}


[[noreturn]] inline void tt_throw(char const* file, int line, char const* assert_type, char const* condition_str) {
    tt_throw_impl(file, line, assert_type, condition_str);
}

template <typename... Args>
[[noreturn]] void tt_throw(
    char const* file,
    int line,
    char const* assert_type,
    char const* condition_str,
    fmt::format_string<Args const&...> fmt,
    Args const&... args) {
    detail::tt_throw_impl(file, line, assert_type, condition_str, fmt, args...);
}

inline void tt_assert(char const* file, int line, char const* assert_type, bool condition, char const* condition_str) {
    if (not condition) {
        tt_throw(file, line, assert_type, condition_str);
    }
}

template <typename... Args>
void tt_assert(
    char const* file,
    int line,
    char const* assert_type,
    bool condition,
    char const* condition_str,
    fmt::format_string<Args const&...> fmt,
    Args const&... args) {
    if (not condition) {
        tt_throw(file, line, assert_type, condition_str, fmt, args...);
    }
}

}

}  // namespace tt::assert

// Adding do while around TT_ASSERT to allow flexible usage of the macro. More details can be found in Stack Overflow
// post:
// https://stackoverflow.com/questions/55933541/else-without-previous-if-error-when-defining-macro-with-arguments/55933720#55933720
#ifdef DEBUG
#ifndef TT_ASSERT
#define TT_ASSERT(condition, ...)                                                                           \
    do {                                                                                                    \
        if (not(condition)) [[unlikely]]                                                                    \
            tt::assert::detail::tt_assert(__FILE__, __LINE__, "TT_ASSERT", (condition), #condition, ##__VA_ARGS__); \
    } while (0)  // NOLINT(cppcoreguidelines-macro-usage)
#endif
#else
#define TT_ASSERT(condition, ...)
#endif

#ifndef TT_THROW
#define TT_THROW(...) tt::assert::detail::tt_throw(__FILE__, __LINE__, "TT_THROW", "tt::exception", ##__VA_ARGS__)
#endif

#ifndef TT_FATAL
#define TT_FATAL(condition, message, ...)                                                             \
    do {                                                                                              \
        if (not(condition)) [[unlikely]] {                                                            \
            tt::assert::detail::tt_throw(__FILE__, __LINE__, "TT_FATAL", #condition, message, ##__VA_ARGS__); \
            __builtin_unreachable();                                                                  \
        }                                                                                             \
    } while (0)  // NOLINT(cppcoreguidelines-macro-usage)
#endif
