// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_path.hpp"

#include <cstdlib>
#include <stdexcept>

namespace ttml::utils {

std::filesystem::path get_data_root() {
    // 1. Check TT_TRAIN_DATA environment variable
    if (const char* tt_train_data = std::getenv("TT_TRAIN_DATA")) {
        return std::filesystem::path(tt_train_data);
    }

    // 2. Check TT_METAL_HOME environment variable
    if (const char* tt_metal_home = std::getenv("TT_METAL_HOME")) {
        return std::filesystem::path(tt_metal_home) / "data";
    }

// 3. Use compile-time macro (set by CMake to CMAKE_SOURCE_DIR/data)
#ifdef TTML_DATA_ROOT
    return std::filesystem::path(TTML_DATA_ROOT);
#else
    throw std::runtime_error(
        "Cannot determine data root. Set TT_TRAIN_DATA or TT_METAL_HOME environment variable, "
        "or rebuild with TTML_DATA_ROOT defined.");
#endif
}

std::filesystem::path resolve_data_path(
    const std::string& data_path, const std::optional<std::filesystem::path>& data_root) {
    std::filesystem::path path(data_path);

    // Absolute paths are returned as-is
    if (path.is_absolute()) {
        return path;
    }

    // Relative paths are joined with data_root
    auto root = data_root.value_or(get_data_root());
    return root / path;
}

std::filesystem::path resolve_tokenizer_path(
    const std::string& tokenizer_path, const std::optional<std::filesystem::path>& tokenizer_root) {
    std::filesystem::path path(tokenizer_path);

    // Absolute paths are returned as-is
    if (path.is_absolute()) {
        return path;
    }

    // Relative paths are joined with tokenizer_root
    auto root = tokenizer_root.value_or(get_data_root());
    return root / path;
}

}  // namespace ttml::utils
