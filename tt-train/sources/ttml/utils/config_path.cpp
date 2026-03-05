// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "config_path.hpp"

#include <cstdlib>
#include <stdexcept>

namespace ttml::utils {

std::filesystem::path get_tt_train_root() {
    // 1. Check TT_TRAIN_ROOT environment variable
    if (const char* tt_train_root = std::getenv("TT_TRAIN_ROOT")) {
        return std::filesystem::path(tt_train_root);
    }

    // 2. Check TT_METAL_HOME environment variable
    if (const char* tt_metal_home = std::getenv("TT_METAL_HOME")) {
        return std::filesystem::path(tt_metal_home) / "tt-train";
    }

// 3. Use compile-time macro (set by CMake to CMAKE_SOURCE_DIR/tt-train)
#ifdef TTML_ROOT
    return std::filesystem::path(TTML_ROOT);
#else
    throw std::runtime_error(
        "Cannot determine tt-train root. Set TT_TRAIN_ROOT or TT_METAL_HOME environment variable, "
        "or rebuild with TTML_ROOT defined.");
#endif
}

std::filesystem::path get_configs_root() {
    return get_tt_train_root() / "configs";
}

std::filesystem::path resolve_config_path(
    const std::string& config_path, const std::optional<std::filesystem::path>& configs_root) {
    std::filesystem::path path(config_path);

    // Absolute paths are returned as-is
    if (path.is_absolute()) {
        return path;
    }

    // Relative paths are joined with configs_root
    auto root = configs_root.value_or(get_configs_root());
    return root / path;
}

std::filesystem::path resolve_training_config(const std::string& config_name) {
    return resolve_config_path(config_name, get_configs_root() / "training_configs");
}

std::filesystem::path resolve_model_config(const std::string& config_name) {
    return resolve_config_path(config_name, get_configs_root() / "model_configs");
}

std::filesystem::path resolve_multihost_config(const std::string& config_name) {
    return resolve_config_path(config_name, get_configs_root() / "multihost_configs");
}

}  // namespace ttml::utils
