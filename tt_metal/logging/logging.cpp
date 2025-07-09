// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::logging {

// TODO: Figure out how to expose this set_level API to consumers.
// The idea here is to allow clients to set log level without using environment variables.
// For instance, they may not want to see info messages at all, but only CRITICAL errors

/**
 * @brief Logging severity levels
 *
 * Defines the different severity levels for logging messages, from most
 * verbose (trace) to most severe (critical), with an option to disable
 * logging completely (off).
 */
enum class level {
    trace,     ///< Most detailed logging level, for tracing program execution
    debug,     ///< Debugging information, useful during development
    info,      ///< General informational messages about program operation
    warn,      ///< Warning messages for potentially harmful situations
    error,     ///< Error messages for serious problems
    critical,  ///< Critical errors that may lead to program termination
    off        ///< Disables all logging
};

/// Map our internal enum to spdlog's level enum.
spdlog::level::level_enum to_spdlog_level(level lvl) {
    switch (lvl) {
        case level::trace: return spdlog::level::trace;
        case level::debug: return spdlog::level::debug;
        case level::info: return spdlog::level::info;
        case level::warn: return spdlog::level::warn;
        case level::error: return spdlog::level::err;
        case level::critical: return spdlog::level::critical;
        case level::off: return spdlog::level::off;
    }
    return spdlog::level::info;  // fallback
}

void set_level(level lvl) { ::tt::LoggerRegistry::instance().set_level(to_spdlog_level(lvl)); }

}  // namespace tt::tt_metal::logging
