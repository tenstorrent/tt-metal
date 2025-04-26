// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <unistd.h>
#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) && (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
#include <pybind11/iostream.h>
#endif

#include <fmt/format.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <spdlog/spdlog.h>

// Forwarding Macros
#define TT_LOG_INFO(fmt, ...) SPDLOG_INFO(fmt, ##__VA_ARGS__)
#define TT_LOG_INFO_WITH_CAT(category, fmt, ...) SPDLOG_INFO("[{}] | " fmt, category, ##__VA_ARGS__)

#define TT_LOG_WARN(fmt, ...) SPDLOG_WARN(fmt, ##__VA_ARGS__)
#define TT_LOG_WARN_WITH_CAT(category, fmt, ...) SPDLOG_WARN("[{}] | " fmt, category, ##__VA_ARGS__)

#define TT_LOG_ERROR(fmt, ...) SPDLOG_ERROR(fmt, ##__VA_ARGS__)
#define TT_LOG_ERROR_WITH_CAT(category, fmt, ...) SPDLOG_ERROR("[{}] | " fmt, category, ##__VA_ARGS__)

#define TT_LOG_DEBUG(fmt, ...) SPDLOG_DEBUG(fmt, ##__VA_ARGS__)
#define TT_LOG_DEBUG_WITH_CAT(category, fmt, ...) SPDLOG_DEBUG("[{}] | " fmt, category, ##__VA_ARGS__)

#define TT_LOG_TRACE(fmt, ...) SPDLOG_TRACE(fmt, ##__VA_ARGS__)
#define TT_LOG_TRACE_WITH_CAT(category, fmt, ...) SPDLOG_TRACE("[{}] | " fmt, category, ##__VA_ARGS__)

namespace tt {

#define LOGGER_TYPES \
    X(Always)        \
    X(Test)          \
    X(Timer)         \
    X(Device)        \
    X(Model)         \
    X(LLRuntime)     \
    X(Loader)        \
    X(IO)            \
    X(CompileTrisc)  \
    X(BuildKernels)  \
    X(Verif)         \
    X(Golden)        \
    X(Op)            \
    X(HLK)           \
    X(HLKC)          \
    X(Reportify)     \
    X(GraphCompiler) \
    X(Dispatch)      \
    X(Fabric)        \
    X(Metal)         \
    X(MetalTrace)

enum LogType : uint32_t {
// clang-format off
#define X(a) Log ## a,
    LOGGER_TYPES
#undef X
    LogType_Count,
    // clang-format on
};
static_assert(LogType_Count < 64, "Exceeded number of log types");

constexpr const char* log_type_names[] = {
#define X(name) #name,
    LOGGER_TYPES
#undef X
};

constexpr const char* logtype_to_string(LogType logtype) {
    return static_cast<std::size_t>(logtype) < std::size(log_type_names)
               ? log_type_names[static_cast<std::size_t>(logtype)]
               : "UnknownType";
}

#pragma GCC visibility push(hidden)
class Logger {
public:
    static constexpr char const* type_names[LogType_Count] = {
    // clang-format off
#define X(a) #a,
      LOGGER_TYPES
#undef X
        // clang-format on
    };

    enum class Level {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warning = 3,
        Error = 4,
        Fatal = 5,

        Count,
    };

    static constexpr char const* level_names[] = {
        "TRACE",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "FATAL",
    };

    static_assert(
        (sizeof(level_names) / sizeof(level_names[0])) == static_cast<std::underlying_type_t<Level>>(Level::Count));

    static constexpr fmt::color level_color[] = {
        fmt::color::gray,
        fmt::color::gray,
        fmt::color::cornflower_blue,
        fmt::color::orange,
        fmt::color::red,
        fmt::color::red,
    };

    static_assert(
        (sizeof(level_color) / sizeof(level_color[0])) == static_cast<std::underlying_type_t<Level>>(Level::Count));

    // TODO: we should sink this into some common cpp file, marking inline so maybe it's per lib instead of per TU
    static inline Logger& get() {
        static Logger logger;
        return logger;
    }

    template <typename... Args>
    void log_level_type(Level level, LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
        if (static_cast<std::underlying_type_t<Level>>(level) < static_cast<std::underlying_type_t<Level>>(min_level)) {
            return;
        }

        if ((1 << type) & mask) {
#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) && (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
            pybind11::scoped_ostream_redirect stream(*fd);
#endif
            const auto level_style =
                fmt::fg(level_color[static_cast<std::underlying_type_t<Level>>(level)]) | fmt::emphasis::bold;
            const auto level_str = fmt::format(
                use_styles ? level_style : fmt::text_style{},
                "{:8}",
                level_names[static_cast<std::underlying_type_t<Level>>(level)]);
            const auto type_style = fmt::fg(fmt::color::green);
            const auto type_str = fmt::format(use_styles ? type_style : fmt::text_style{}, "{:>23}", type_names[type]);
            fmt::print(*fd, "{} | {} | ", type_str, level_str);
            fmt::print(*fd, fmt, std::forward<Args>(args)...);
            *fd << std::endl;
        }
    }

    void flush() { *fd << std::flush; }

    bool get_use_styles() const { return this->use_styles; }

private:
    Logger() {
        static char const* env = std::getenv("TT_METAL_LOGGER_TYPES");
        if (env) {
            if (strstr(env, "All")) {
                mask = 0xFFFFFFFFFFFFFFFF;
            } else {
                std::uint32_t mask_index = 0;
                for (char const* type_name : type_names) {
                    mask |= (strstr(env, type_name) != nullptr) << mask_index;
                    mask_index++;
                }
            }
        } else {
            // For now default to all
            mask = 0xFFFFFFFFFFFFFFFF;
        }

        static char const* level_env = std::getenv("TT_METAL_LOGGER_LEVEL");
        if (level_env) {
            std::string level_str = level_env;
            std::transform(
                level_str.begin(), level_str.end(), level_str.begin(), [](unsigned char c) { return std::toupper(c); });
            std::underlying_type_t<Level> level_index = 0;
            for (char const* level_name : level_names) {
                if (level_str == level_name) {
                    min_level = static_cast<Level>(level_index);
                }
                level_index++;
            }
        }

        use_styles = isatty(fileno(stdout));

#if !defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) || (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 0)
        static char const* file_env = std::getenv("TT_METAL_LOGGER_FILE");
        if (file_env) {
            log_file.open(file_env);
            if (log_file.is_open()) {
                fd = &log_file;
                use_styles = false;
            }
        }
#endif
    }

    std::ofstream log_file;
    std::ostream* fd = &std::cout;
    std::uint64_t mask = (1 << LogAlways);
    Level min_level = Level::Info;
    bool use_styles = true;
};
#pragma GCC visibility pop

#ifdef DEBUG
template <typename... Args>
static void log_debug(LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get().log_level_type(Logger::Level::Debug, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_debug(fmt::format_string<Args...> fmt, Args&&... args) {
    log_debug(LogAlways, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_trace_(
    LogType type,
    const fmt::text_style src_style,
    const std::string& src_info,
    fmt::format_string<const std::string&, Args...> fmt,
    Args&&... args) {
    const bool use_styles = Logger::get().get_use_styles();
    const auto src_info_stylized = use_styles ? fmt::format(src_style, fmt::runtime(src_info)) : src_info;
    Logger::get().log_level_type(Logger::Level::Trace, type, fmt, src_info_stylized, std::forward<Args>(args)...);
}

#define log_trace(log_type, ...) \
    log_trace_(log_type, fmt::fg(fmt::color::green), fmt::format("{}:{}", __FILE__, __LINE__), "{} - " __VA_ARGS__)
#else
template <typename... Args>
static void log_debug(LogType type, fmt::format_string<Args...> fmt, Args&&... args) {}
template <typename... Args>
static void log_debug(fmt::format_string<Args...> fmt, Args&&... args) {}
#define log_trace(...) ((void)0)
#endif

template <typename... Args>
static void log(Logger::Level log_level, LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get().log_level_type(log_level, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_warning(LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get().log_level_type(Logger::Level::Warning, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_warning(fmt::format_string<Args...> fmt, Args&&... args) {
    log_warning(LogAlways, fmt, std::forward<Args>(args)...);
}

static void log_warning(char const* str) { log_warning(LogAlways, "{}", str); }

template <typename... Args>
static void log_error(LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get().log_level_type(Logger::Level::Error, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_error(fmt::format_string<Args...> fmt, Args&&... args) {
    log_error(LogAlways, fmt, std::forward<Args>(args)...);
}

static void log_error(char const* str) { log_error(LogAlways, "{}", str); }

template <typename... Args>
static void log_fatal(LogType type, fmt::format_string<Args...> fmt, Args&&... args) {
    Logger::get().log_level_type(Logger::Level::Fatal, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_fatal(fmt::format_string<Args...> fmt, Args&&... args) {
    log_fatal(LogAlways, fmt, std::forward<Args>(args)...);
}

static void log_fatal(char const* str) { log_fatal(LogAlways, "{}", str); }

#undef LOGGER_TYPES

}  // namespace tt

// Custom formatter for LogType
namespace fmt {
template <>
struct fmt::formatter<tt::LogType> : fmt::formatter<std::string_view> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    constexpr auto format(tt::LogType logtype, FormatContext& ctx) const {
        return fmt::formatter<std::string_view>::format(tt::logtype_to_string(logtype), ctx);
    }
};
}  // namespace fmt
