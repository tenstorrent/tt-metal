// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace ttml::utils {

/**
 * Get the data root directory.
 *
 * Resolution order:
 * 1. TT_TRAIN_DATA environment variable (if set)
 * 2. TT_METAL_HOME/data (if TT_METAL_HOME is set)
 * 3. TTML_DATA_ROOT compile-time macro (set by CMake to CMAKE_SOURCE_DIR/data)
 *
 * @return Path to data root directory
 * @throws std::runtime_error if data root cannot be determined
 */
std::filesystem::path get_data_root();

/**
 * Resolve a data path. If the path is:
 * - Absolute: return as-is
 * - Relative: join with the provided data_root (or get_data_root() if not specified)
 *
 * @param data_path Data file name or relative path (e.g., "shakespeare.txt" or "tokenized/data.csv")
 * @param data_root Optional root directory for data resolution
 * @return Resolved absolute path to data file
 */
std::filesystem::path resolve_data_path(
    const std::string& data_path, const std::optional<std::filesystem::path>& data_root = std::nullopt);

std::filesystem::path resolve_tokenizer_path(
    const std::string& tokenizer_path, const std::optional<std::filesystem::path>& tokenizer_root = std::nullopt);
}  // namespace ttml::utils
// namespace ttml::utils
