// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file logging.cpp
 * @brief Implementation of Metal logging initialization
 *
 * This file contains the initialization code for the Metal logging system.
 * It creates a static instance of the TT LoggerInitializer with specific
 * environment variable names for Metal logging configuration.
 */

#include <tt-logger/tt-logger-initializer.hpp>

namespace tt::tt_metal::logging {

constexpr auto file_env_var = "TT_METAL_LOGGER_FILE";
constexpr auto level_env_var = "TT_METAL_LOGGER_LEVEL";
constexpr auto log_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v";

/**
 * @brief Static instance of LoggerInitializer for Metal logging
 *
 * This static instance initializes the logging system with Metal-specific
 * environment variables:
 * - TT_METAL_LOGGER_FILE: Controls the log file path for Metal logging
 * - TT_METAL_LOGGER_LEVEL: Controls the log level for Metal logging
 *
 * The logger will be initialized when this translation unit is loaded,
 * setting up either file-based or console-based logging depending on
 * the environment variable configuration.
 */
static tt::LoggerInitializer loggerInitializer(file_env_var, level_env_var, log_pattern);

}  // namespace tt::tt_metal::logging
