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

#include "tt_metal/common/logger.hpp"

namespace tt {
template <typename A, typename B>
struct OStreamJoin {
    OStreamJoin(A const& a, B const& b, char const* delim = " ") : a(a), b(b), delim(delim) {}
    A const& a;
    B const& b;
    char const* delim;
};

template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, tt::OStreamJoin<A, B> const& join) {
    os << join.a << join.delim << join.b;
    return os;
}
}  // namespace tt

namespace tt::assert {

static std::string demangle(const char* str) {
    size_t size = 0;
    int status = 0;
    std::string rt(256, '\0');
    if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", &rt[0])) {
        char* v = abi::__cxa_demangle(&rt[0], nullptr, &size, &status);
        if (v) {
            std::string result(v);
            free(v);
            return result;
        }
    }
    return str;
}

// https://www.fatalerrors.org/a/backtrace-function-and-assert-assertion-macro-encapsulation.html

/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
inline std::vector<std::string> backtrace(int size = 64, int skip = 1) {
    std::vector<std::string> bt;
    void** array = (void**)malloc((sizeof(void*) * size));
    size_t s = ::backtrace(array, size);
    char** strings = backtrace_symbols(array, s);
    if (strings == NULL) {
        std::cout << "backtrace_symbols error." << std::endl;
        return bt;
    }
    for (size_t i = skip; i < s; ++i) {
        bt.push_back(demangle(strings[i]));
    }
    free(strings);
    free(array);

    return bt;
}

/**
 * @brief String to get current stack information
 * @param[in] size Maximum number of stacks
 * @param[in] skip Skip the number of layers at the top of the stack
 * @param[in] prefix Output before stack information
 */
inline std::string backtrace_to_string(int size = 64, int skip = 2, const std::string& prefix = "") {
    std::vector<std::string> bt = backtrace(size, skip);
    std::stringstream ss;
    for (size_t i = 0; i < bt.size(); ++i) {
        ss << prefix << bt[i] << std::endl;
    }
    return ss.str();
}

template <typename... Args>
[[noreturn]] void tt_throw_impl(
    char const* file, int line, char const* assert_type, char const* condition_str, Args const&... args) {
    std::stringstream trace_message_ss = {};
    trace_message_ss << assert_type << " @ " << file << ":" << line << ": " << condition_str << std::endl;
    if constexpr (sizeof...(args) > 0) {
        trace_message_ss << "info:" << std::endl;
        trace_message_ss << fmt::format(args...) << std::endl;
        log_fatal(args...);
    }
    trace_message_ss << "backtrace:\n";
    trace_message_ss << tt::assert::backtrace_to_string(100, 3, " --- ");
    trace_message_ss << std::flush;
    Logger::get().flush();
    if (std::getenv("TT_ASSERT_ABORT"))
        abort();
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
    tt_throw_impl(file, line, assert_type, condition_str, fmt, args...);
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

}  // namespace tt::assert

// Adding do while around TT_ASSERT to allow flexible usage of the macro. More details can be found in Stack Overflow
// post:
// https://stackoverflow.com/questions/55933541/else-without-previous-if-error-when-defining-macro-with-arguments/55933720#55933720
#ifdef DEBUG
#ifndef TT_ASSERT
#define TT_ASSERT(condition, ...)                                                                           \
    do {                                                                                                    \
        if (not(condition)) [[unlikely]]                                                                    \
            tt::assert::tt_assert(__FILE__, __LINE__, "TT_ASSERT", (condition), #condition, ##__VA_ARGS__); \
    } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif
#else
#define TT_ASSERT(condition, ...)
#endif

#ifndef TT_THROW
#define TT_THROW(...) tt::assert::tt_throw(__FILE__, __LINE__, "TT_THROW", "tt::exception", ##__VA_ARGS__)
#endif

#ifndef TT_FATAL
#define TT_FATAL(condition, message, ...)                                                             \
    do {                                                                                              \
        if (not(condition)) [[unlikely]] {                                                            \
            tt::assert::tt_throw(__FILE__, __LINE__, "TT_FATAL", #condition, message, ##__VA_ARGS__); \
            __builtin_unreachable();                                                                  \
        }                                                                                             \
    } while (0) // NOLINT(cppcoreguidelines-macro-usage)
#endif
